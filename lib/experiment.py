"""Generic CPE experiment runner: data -> factor training -> successive-halving
selection on val -> top-1 vs baseline on test -> artifacts into results/<run>/
(synced back from remote clusters via the scaffold's instance.sync).

An experiment supplies only its data (make_split) and its scoring (score_fn +
which metric selects). Everything else is shared.
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

from lib.generation import build_prompts, generate_completions
from lib.methods import STEER_METHODS
from lib.selection import successive_halving

SCORE_CONCURRENCY = int(os.environ.get("CPE_SCORE_CONCURRENCY", "32"))


def _score_batch(score_fn, pairs):
    """Score (completion, answer) pairs concurrently. Instant for local scorers;
    essential for network judge scorers (jailbreak). Preserves order."""
    with ThreadPoolExecutor(max_workers=SCORE_CONCURRENCY) as pool:
        return list(pool.map(lambda p: score_fn(*p), pairs))


def _logged(comp):
    """Completion fields for the log: token ids travel with the text when the
    scorer needed them, so post-hoc analysis can re-read the same channel."""
    if isinstance(comp, str):
        return {'completion': comp}
    return {'completion': comp['text'], 'token_ids': comp['token_ids']}


def _generate_labeled(make_split, score_fn, primary_metric, gold_fn, anti_gold_fn, cfg,
                      tokenizer, system_prompt, enable_thinking, model_name, max_new_tokens,
                      backend, max_model_len, tensor_parallel, out_path, base_adapter):
    """Supervised signal for sft (correct only) and diffmeans (both).

    gold_only: clean dataset targets — correct = gold_fn(answer), incorrect =
    anti_gold_fn(answer), one per train prompt (no model generation). Otherwise:
    base-model completions split by primary_metric (STaR), sampled
    completions_per_prompt times, capped per class."""
    prompts, answers = make_split("train", cfg['num_prompts'])
    chat = build_prompts(tokenizer, prompts, system_prompt, enable_thinking)
    if cfg.get('gold_only'):
        # sft needs only correct gold targets; diffmeans also needs the wrong-answer
        # gold (anti_gold_fn). Envs without a single wrong answer (countdown) pass
        # anti_gold_fn=None and use gold only for sft.
        labeled = {'correct': [{'prompt': p, 'completion': gold_fn(a)}
                               for p, a in zip(chat, answers)],
                   'incorrect': [{'prompt': p, 'completion': anti_gold_fn(a)}
                                 for p, a in zip(chat, answers)] if anti_gold_fn else []}
        with open(out_path, 'w') as f:
            json.dump(labeled, f, indent=2)
        print(f"Labeled (gold): {len(labeled['correct'])} correct / "
              f"{len(labeled['incorrect'])} incorrect gold targets")
        return labeled
    rep = cfg['completions_per_prompt']
    chat_rep = [c for c in chat for _ in range(rep)]
    ans_rep = [a for a in answers for _ in range(rep)]
    comps = generate_completions(model_name, {'base': base_adapter}, chat_rep, max_new_tokens,
                                 cfg['temperature'], backend, max_model_len,
                                 tensor_parallel=tensor_parallel)['base']
    scored = _score_batch(score_fn, list(zip(comps, ans_rep)))
    correct, incorrect = [], []
    for prompt, comp, metrics in zip(chat_rep, comps, scored):
        bucket = correct if metrics[primary_metric] else incorrect
        bucket.append({'prompt': prompt, 'completion': comp})
    if cfg['include_gold']:
        correct += [{'prompt': p, 'completion': gold_fn(a)} for p, a in zip(chat, answers)]
    labeled = {'correct': correct[:cfg['max_correct']],
               'incorrect': incorrect[:cfg['max_incorrect']]}
    with open(out_path, 'w') as f:
        json.dump(labeled, f, indent=2)
    print(f"Labeled: {len(labeled['correct'])} correct, {len(labeled['incorrect'])} incorrect")
    return labeled


def run_cpe_experiment(
    make_split,            # (split, n) -> (instructions, answers)
    score_fn,              # (completion, answer) -> dict of metrics
    primary_metric: str,   # metric used for selection + headline test number
    log_path: str,
    model_name: str,
    system_prompt: str,
    enable_thinking: bool,
    num_train_prompts: int,
    num_val_prompts: int,
    num_test_prompts: int,
    source_layers: list,
    target_layer: int,
    num_factors: int,
    num_iters: int,
    factor_batch_size: int,
    max_seq_len: int,
    train_seed: int,
    device: str,
    model_dtype: str,
    trim: bool,
    generation_backend: str,
    max_new_tokens: int,
    temperature: float,
    max_model_len: int,
    tensor_parallel: int,
    selection_schedule: list,
    method: str,
    sae_config,
    return_token_ids: bool,
    target_modules: list,
    model_class: str,
    vllm_additional_config: dict,
    label_config=None,     # method-specific (sft/diffmeans); None for cpe/random/sae
    sft_config=None,
    steer_config=None,
    gold_fn=None,          # answer -> gold (correct) completion text
    anti_gold_fn=None,     # answer -> gold WRONG completion text (gold_only diffmeans)
    base_adapter=None,     # always-on PEFT adapter dir (the organism). Kept in the
                           # LoRA path everywhere — hooks during training, rank-
                           # concatenation for generation — never merged into the
                           # weights, where bf16 rounding erases the delta (nb 004).
):
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)

    # === data ===
    train_prompts, _ = make_split("train", num_train_prompts)
    val_prompts, val_answers = make_split("val", num_val_prompts)
    test_prompts, test_answers = make_split("test", num_test_prompts)

    # === factor training (subprocess: process exit releases all GPU memory,
    # so the vLLM engine afterwards starts on a clean device) ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
    train_chat = build_prompts(tokenizer, train_prompts, system_prompt, enable_thinking)
    token_ids = [tokenizer.encode(p, truncation=True, max_length=max_seq_len,
                                  add_special_tokens=False) for p in train_chat]
    adapter_root = os.path.abspath(os.path.join(log_path, "adapters"))

    # supervised baselines: label base-model completions before training
    labeled = None
    if method in ("sft", "diffmeans"):
        labeled = _generate_labeled(
            make_split, score_fn, primary_metric, gold_fn, anti_gold_fn, label_config,
            tokenizer, system_prompt, enable_thinking, model_name, max_new_tokens,
            generation_backend, max_model_len, tensor_parallel,
            os.path.join(log_path, "labeled_completions.json"), base_adapter)

    train_args = dict(
        model_name=model_name, model_dtype=model_dtype, device=device,
        token_ids=token_ids, method=method, sae_config=sae_config,
        target_modules=target_modules, model_class=model_class,
        source_layers=source_layers, target_layer=target_layer,
        num_factors=num_factors, num_iters=num_iters,
        factor_batch_size=factor_batch_size, train_seed=train_seed, trim=trim,
        adapter_root=adapter_root, labeled=labeled, sft_config=sft_config,
        steer_config=steer_config, base_adapter=base_adapter,
        steer_path=os.path.abspath(os.path.join(log_path, "steer_vectors.pt")),
        log_dir=os.path.abspath(os.path.join(log_path, "training")),
    )
    args_path = os.path.abspath(os.path.join(log_path, "train_args.json"))
    with open(args_path, 'w') as f:
        json.dump(train_args, f)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run([sys.executable, '-m', 'lib.train_proc', args_path],
                   check=True, cwd=repo_root, env=env)

    # steering methods return vectors applied as forward hooks (lib/steer_hooks),
    # exactly, rather than the o_proj LoRA encoding they used to be squeezed into.
    # Candidates then differ only by `steer`; the adapter is the organism throughout.
    steer, steer_site = None, "layer"
    if method in STEER_METHODS:
        payload = torch.load(train_args['steer_path'], weights_only=True)
        steer, steer_site = payload['candidates'], payload['site']
        adapters = {name: base_adapter for name in steer}
        print(f"Loaded {len(steer)} steering candidates, site={steer_site}")
    else:
        # sae may produce fewer factors than requested: read back what was exported
        adapters = {name: os.path.join(adapter_root, name)
                    for name in sorted(os.listdir(adapter_root),
                                       key=lambda n: int(n.split('_')[1]))}
    if base_adapter and steer is None:
        from lib.compose import compose_adapters
        # under tmp/, not log_path: composed adapters are ~140MB each and must
        # stay out of the results folder that instance.sync rsyncs back
        composed_root = os.path.abspath(os.path.join(
            log_path, "..", "..", "tmp",
            "composed_" + os.path.basename(os.path.normpath(log_path))))
        adapters = {name: compose_adapters(base_adapter, path,
                                           os.path.join(composed_root, name))
                    for name, path in adapters.items()}
        print(f"Composed {len(adapters)} factor adapters with the base adapter")
    if torch.cuda.is_available():
        print(f"CUDA free after training: "
              f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB")

    # === successive-halving selection on val ===
    val_chat = build_prompts(tokenizer, val_prompts, system_prompt, enable_thinking)
    completion_log = []

    def eval_fn(candidates, prompt_indices):
        subset = {name: adapters[name] for name in candidates}
        prompts = [val_chat[i] for i in prompt_indices]
        completions = generate_completions(
            model_name, subset, prompts, max_new_tokens, temperature,
            generation_backend, max_model_len, tensor_parallel=tensor_parallel,
            return_token_ids=return_token_ids,
            additional_config=vllm_additional_config,
            steer=None if steer is None else {n: steer[n] for n in candidates},
            steer_site=steer_site)
        flat = [(name, pidx, comp)
                for name, comps in completions.items()
                for pidx, comp in zip(prompt_indices, comps)]
        metrics_all = _score_batch(score_fn, [(c, val_answers[p]) for _, p, c in flat])
        out = {name: {} for name in completions}
        for (name, pidx, comp), metrics in zip(flat, metrics_all):
            out[name][pidx] = float(metrics[primary_metric])
            completion_log.append({'split': 'val', 'factor': name,
                                   'prompt_idx': pidx, **_logged(comp), **metrics})
        return out

    selection = successive_halving(
        list(adapters), num_prompts=len(val_prompts),
        schedule=[tuple(s) for s in selection_schedule], eval_fn=eval_fn)
    best_factor = selection['ranking'][0][0]

    # === test: top-1 vs no-adapter baseline ===
    test_chat = build_prompts(tokenizer, test_prompts, system_prompt, enable_thinking)
    test_completions = generate_completions(
        model_name, {'baseline': base_adapter, best_factor: adapters[best_factor]},
        test_chat, max_new_tokens, temperature, generation_backend,
        max_model_len, tensor_parallel=tensor_parallel,
        return_token_ids=return_token_ids,
        additional_config=vllm_additional_config,
        # {} = this arm gets no hooks: the baseline is the unsteered organism
        steer=None if steer is None else {'baseline': {}, best_factor: steer[best_factor]},
        steer_site=steer_site)
    test_results = {}
    for name, comps in test_completions.items():
        metrics_list = _score_batch(score_fn, [(c, test_answers[i])
                                               for i, c in enumerate(comps)])
        for pidx, (comp, metrics) in enumerate(zip(comps, metrics_list)):
            completion_log.append({'split': 'test', 'factor': name,
                                   'prompt_idx': pidx, **_logged(comp), **metrics})
        n = len(metrics_list)
        test_results[name] = {k: sum(float(m[k]) for m in metrics_list) / n
                              for k in metrics_list[0]}
        test_results[name]['n_prompts'] = n

    # === persist ===
    with open(os.path.join(log_path, "selection.json"), 'w') as f:
        json.dump({'ranking': selection['ranking'], 'rounds': selection['rounds'],
                   'scores': selection['scores']}, f, indent=2)
    with open(os.path.join(log_path, "completions.json"), 'w') as f:
        json.dump(completion_log, f, indent=2)
    with open(os.path.join(log_path, "test_results.json"), 'w') as f:
        json.dump({'best_factor': best_factor, 'primary_metric': primary_metric,
                   'results': test_results}, f, indent=2)

    print(f"Best factor: {best_factor}")
    print(f"Test results: {json.dumps(test_results, indent=2)}")
