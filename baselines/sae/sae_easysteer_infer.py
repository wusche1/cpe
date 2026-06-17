"""SAE-steering baseline via EasySteer (forked vLLM, native steering). For each SAE
feature (decoder vector from sae_select, ranked by ||J w||), steer the residual at
the single source site (resid_post of the final source block) by `scale_mult * norm`
and generate the env's prompts. Emits CPE-format inference JSON (factor_idx = SAE
feature index) so the EXISTING CPE scoring stage scores it identically to CPE
(cpe.pipeline --stage score). Single layer only.

Sharded: pass --start/--end to steer a contiguous slice of the ranked feature list
on one GPU; merge shards afterward into inference/inference_results.json.

Run with the EasySteer env: .venv_sae/bin/python (has the vllm-steer fork).
Usage (one shard):
  sae_easysteer_infer.py --env convo_llama --scale_mult 0.2 --split val \
    --start 0 --end 64 --n 100 --out outputs/sae_convo_llama/inference/shard_0.json
"""
import argparse
import json
import os
import tempfile

import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# env -> (cpe config, selection/output dir). Selection (selection.json +
# steering_vectors.pt) and merged inference live under the same dir.
ENVS = {
    "countdown_llama":  ("configs/countdown_llama.json",  "outputs/sae_countdown_llama"),
    "countdown_qwen":   ("configs/countdown_qwen.json",   "outputs/sae_countdown_qwen"),
    "sycophancy_llama": ("configs/sycophancy_llama.json", "outputs/sae_sycophancy_llama"),
    "sycophancy_qwen":  ("configs/sycophancy_qwen.json",  "outputs/sae_sycophancy_qwen"),
    "jailbreak_llama":  ("configs/jailbreak_llama.json",  "outputs/sae_jailbreak_llama"),
    "convo_llama":      ("configs/convo_llama.json",       "outputs/sae_convo_llama"),
    "convo_qwen":       ("configs/convo_qwen.json",        "outputs/sae_convo_qwen"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(ENVS))
    ap.add_argument("--scale_mult", type=float, default=0.2, help="multiple of avg residual norm")
    ap.add_argument("--split", default="val", help="literal dataset split dir name (val/test)")
    ap.add_argument("--start", type=int, default=0, help="first ranked feature index (inclusive)")
    ap.add_argument("--end", type=int, default=512, help="last ranked feature index (exclusive)")
    ap.add_argument("--n", type=int, default=100, help="num prompts from the split (first n)")
    ap.add_argument("--max_model_len", type=int, default=0)
    ap.add_argument("--baseline", action="store_true", help="also emit factor_idx=-1 unsteered baseline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    a = ap.parse_args()

    cfg_path, sae_dir = ENVS[a.env]
    cfg = json.load(open(cfg_path))
    sel = json.load(open(f"{sae_dir}/selection.json"))
    vecs = torch.load(f"{sae_dir}/steering_vectors.pt", map_location="cpu", weights_only=False).float()  # (m,d_model) unit
    norm = float(sel["steering_norm"])
    steer_layer = int(sel["steer_site_hs_idx"]) - 1   # residual after this decoder block
    scale = a.scale_mult * norm
    feats = sel["feature_indices"]
    end = min(a.end, vecs.shape[0])
    print(f"[sae-es:{a.env}:{a.split}:{a.start}-{end}] scale_mult={a.scale_mult} -> scale={scale:.3f} "
          f"(norm={norm:.3f}) layer={steer_layer} model={cfg['model_name']}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds = load_from_disk(os.path.join(cfg["dataset_path"], a.split))
    n = min(a.n, len(ds))
    sysmsg = cfg.get("system_prompt", "")
    prompts, pidx = [], []
    for i in range(n):
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
               [{"role": "user", "content": ds[i][cfg["prompt_field"]]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                       enable_thinking=cfg.get("enable_thinking", False)))
        pidx.append(i)

    mml = a.max_model_len or cfg.get("max_model_len", 2048)
    llm = LLM(model=cfg["model_name"], enable_steer_vector=True, enforce_eager=True,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem,
              max_model_len=mml, trust_remote_code=True)
    sp = SamplingParams(temperature=cfg.get("temperature", 0.0), top_p=cfg.get("top_p", 1.0),
                        max_tokens=cfg.get("max_tokens", 512),
                        repetition_penalty=cfg.get("repetition_penalty", 1.0))

    tmp = tempfile.mkdtemp(prefix="saevec_")
    results = []

    def emit(fid, name, req):
        out = llm.generate(prompts, steer_vector_request=req, sampling_params=sp)
        results.append({"factor_idx": fid, "adapter_name": name,
                        "responses": [{"prompt_idx": pidx[j], "prompt": ds[pidx[j]][cfg["prompt_field"]],
                                       "response": out[j].outputs[0].text} for j in range(len(prompts))]})

    if a.baseline:
        emit(-1, "baseline", SteerVectorRequest("base", 1, steer_vector_local_path=_save(vecs[0], tmp, "b"),
                                                scale=0.0, target_layers=[steer_layer],
                                                prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1]))
    for r in range(a.start, end):
        fid = int(feats[r])
        req = SteerVectorRequest(f"f{fid}", (r - a.start) + 2, steer_vector_local_path=_save(vecs[r], tmp, str(fid)),
                                 scale=scale, target_layers=[steer_layer],
                                 prefill_trigger_tokens=[-1], generate_trigger_tokens=[-1])
        emit(fid, f"sae_{fid}", req)
        if (r - a.start + 1) % 8 == 0:
            print(f"  steered {r - a.start + 1}/{end - a.start}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"metadata": {"env": a.env, "split": a.split, "scale_mult": a.scale_mult, "scale": scale,
                            "norm": norm, "steer_layer": steer_layer, "start": a.start, "end": end},
               "results": results}, open(a.out, "w"))
    print(f"[sae-es:{a.env}:{a.split}:{a.start}-{end}] wrote {a.out} ({len(results)} entries)", flush=True)


def _save(vec, tmp, name):
    p = os.path.join(tmp, f"{name}.pt")
    torch.save(vec.contiguous(), p)
    return p


if __name__ == "__main__":
    main()
