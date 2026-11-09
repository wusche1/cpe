"""Minimal GRPO (RLVR) with optional steered ("fuzzed") rollouts.

Policy = a trainable LoRA on top of base weights (+ an optional organism LoRA kept
as forward hooks, never merged: bf16 merging erases small deltas, notebook 004).
Rollouts: vLLM, one request LoRA per rollout = rank-concatenation of organism +
policy (+ steering factor), rebuilt after every optimizer step. Steering factors
are rank-1 o_proj LoRAs (CPE FactorSet exports); in the HF forward they ride as
forward hooks (lib/lora_hooks), so both sides apply the identical delta.

backward_steered=True: the training forward keeps the rollout's factor active
(steered samples are on-policy for the steered model). False: the unsteered
policy is trained on them, the PPO ratio pi_unsteered / pi_rollout is then a
genuine off-policy importance ratio and the clip does real work.
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from lib.compose import compose_adapters
from lib.generation import _vllm_lora_rank, build_prompts
from lib.lora_hooks import attach_lora


def group_advantages(rewards, group_size, scale_std, eps):
    r = torch.as_tensor(rewards, dtype=torch.float32).view(-1, group_size)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale_std:
        adv = adv / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)


def grpo_loss(logp, old_logp, adv, mask, clip_low, clip_high, ref_logp=None, kl_coef=0.0):
    """Per-token clipped surrogate (+ k3 KL to ref). All (B, T) except adv (B,).
    Returns the token-SUM of the loss (caller normalises) and per-sample stat sums."""
    ratio = torch.exp(logp - old_logp)
    a = adv.unsqueeze(1)
    unclipped = ratio * a
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high) * a
    loss = -torch.minimum(unclipped, clipped)
    kl = torch.zeros_like(logp)
    if ref_logp is not None:
        d = ref_logp - logp
        kl = torch.exp(d) - d - 1
        loss = loss + kl_coef * kl
    r = ratio.detach()
    big = torch.finfo(r.dtype).max
    stats = {'tokens': mask.sum(1), 'ratio_sum': (r * mask).sum(1),
             'clipped': ((clipped < unclipped).float() * mask).sum(1),
             'kl_sum': (kl.detach() * mask).sum(1),
             'ratio_max': torch.where(mask > 0, r, -big).max(1).values,
             'ratio_min': torch.where(mask > 0, r, big).min(1).values}
    return (loss * mask).sum(), stats


def completion_logprobs(model, prompt_ids, completion_ids, temperature):
    """Log-probs of each completion token, (B, Tc) right-aligned with mask.
    Left padding so all completions end at the last position: only the last Tc+1
    logits are materialised (vocab-sized logits are the memory hog)."""
    device = next(model.parameters()).device
    seqs = [p + c for p, c in zip(prompt_ids, completion_ids)]
    L, Tc = max(map(len, seqs)), max(map(len, completion_ids))
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    attn = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        attn[i, L - len(s):] = 1
    mask = torch.zeros(len(seqs), Tc)
    for i, c in enumerate(completion_ids):
        mask[i, Tc - len(c):] = 1
    ids, attn, mask = ids.to(device), attn.to(device), mask.to(device)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=attn, position_ids=pos,
                   logits_to_keep=Tc + 1).logits[:, :-1].float() / temperature
    logp = torch.gather(F.log_softmax(logits, -1), 2, ids[:, -Tc:, None]).squeeze(-1)
    return logp * mask, mask


def prepare_factors(factor_dirs, scale, random_seed, out_root):
    """Materialise steering factors as adapter dirs with B scaled by `scale`.
    random_seed not None: each factor is replaced by a random one with the same
    keys/shapes and the same per-tensor norms (random-direction control)."""
    out = []
    for i, d in enumerate(factor_dirs):
        sd = load_file(os.path.join(d, "adapter_model.safetensors"))
        gen = torch.Generator().manual_seed(random_seed + i) if random_seed is not None else None
        new = {}
        for k, t in sd.items():
            t = t.float()
            if gen is not None:
                t = F.normalize(torch.randn(t.shape, generator=gen).flatten(), dim=0
                                ).view(t.shape) * t.norm()
            new[k] = (t * scale if "lora_B" in k else t).contiguous()
        dst = os.path.join(out_root, f"factor_{i}")
        os.makedirs(dst, exist_ok=True)
        save_file(new, os.path.join(dst, "adapter_model.safetensors"))
        with open(os.path.join(d, "adapter_config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(dst, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        out.append(dst)
    return out


class Policy:
    def __init__(self, model_name, model_class, organism_adapter, lora_config, lr, device, dtype,
                 gradient_checkpointing):
        import transformers
        from peft import LoraConfig, get_peft_model

        model = getattr(transformers, model_class).from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map=device)
        if organism_adapter:
            attach_lora(model, organism_adapter)
        self.model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora_config))
        # fp32 master copy of the adapter: bf16 Adam updates round away at RL lrs
        params = [p for p in self.model.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

    @contextmanager
    def steered(self, factor_dir):
        handles = attach_lora(self.model.base_model.model, factor_dir) if factor_dir else []
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def save(self, out_dir):
        self.model.save_pretrained(out_dir)
        return out_dir


def train_step(policy, rollouts, backward_steered, temperature, micro_batch_size,
               clip_low, clip_high, kl_coef, max_grad_norm):
    """One optimizer step on rollouts (dicts: prompt_ids, completion_ids, old_logp,
    adv, factor). Token-level normalisation over the whole batch. Zero-advantage
    samples are skipped when there is no KL term (their gradient is exactly 0)."""
    model = policy.model
    model.train()
    n_tok = sum(len(r['completion_ids']) for r in rollouts)
    key = lambda r: (r['factor'] or "") if backward_steered else ""
    todo = sorted([r for r in rollouts if kl_coef > 0 or r['adv'] != 0], key=key)
    stats = []
    i = 0
    while i < len(todo):
        mb = [todo[i]]
        while len(mb) < micro_batch_size and i + len(mb) < len(todo) \
                and key(todo[i + len(mb)]) == key(mb[0]):
            mb.append(todo[i + len(mb)])
        i += len(mb)
        pids, cids = [r['prompt_ids'] for r in mb], [r['completion_ids'] for r in mb]
        with policy.steered(mb[0]['factor'] if backward_steered else None):
            logp, mask = completion_logprobs(model, pids, cids, temperature)
            Tc = mask.shape[1]
            old = torch.zeros_like(logp)
            for j, r in enumerate(mb):
                old[j, Tc - len(r['old_logp']):] = torch.tensor(r['old_logp'])
            ref = None
            if kl_coef > 0:
                with torch.no_grad(), model.disable_adapter():
                    ref, _ = completion_logprobs(model, pids, cids, temperature)
            adv = torch.tensor([r['adv'] for r in mb], device=logp.device)
            loss, s = grpo_loss(logp, old, adv, mask, clip_low, clip_high, ref, kl_coef)
            (loss / n_tok).backward()
        stats += [{k: v[j].item() for k, v in s.items()} | {'steered': r['factor'] is not None}
                  for j, r in enumerate(mb)]
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.params, max_grad_norm).item()
    policy.opt.step()
    policy.opt.zero_grad(set_to_none=True)
    return stats, grad_norm


class VLLMEngine:
    """vLLM rollouts; each request carries organism+policy(+factor) as one LoRA."""

    def __init__(self, model_name, organism_adapter, lora_rank, work_dir, max_loras,
                 gpu_memory_utilization, max_model_len, sleep, seed, additional_config,
                 max_num_seqs):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        torch.cuda.mem_get_info()
        from vllm import LLM
        org_rank = 0
        if organism_adapter:
            with open(os.path.join(organism_adapter, "adapter_config.json")) as f:
                org_rank = json.load(f)['r']
        self.llm = LLM(model=model_name, enable_lora=True,
                       max_lora_rank=_vllm_lora_rank(org_rank + lora_rank + 1),
                       max_loras=max_loras, max_cpu_loras=4 * max_loras,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enable_sleep_mode=sleep,
                       logprobs_mode="processed_logprobs", seed=seed, trust_remote_code=True,
                       additional_config=additional_config, max_num_seqs=max_num_seqs)
        self.organism, self.work_dir, self.sleep = organism_adapter, work_dir, sleep
        self.next_id, self.version = 1, 0

    def update(self, policy):
        self.version += 1
        pol = policy.save(os.path.join(self.work_dir, "policy"))
        base = (compose_adapters(self.organism, pol, os.path.join(self.work_dir, "base"))
                if self.organism else pol)
        self.adapters = {None: base}

    def _request(self, factor):
        from vllm.lora.request import LoRARequest
        if factor not in self.adapters:
            self.adapters[factor] = compose_adapters(
                self.adapters[None], factor,
                os.path.join(self.work_dir, "steered", os.path.basename(factor)))
        req = LoRARequest(f"v{self.version}_{factor}", self.next_id, self.adapters[factor])
        self.next_id += 1
        return req

    def generate(self, prompts, factors, temperature, max_tokens):
        from vllm import SamplingParams
        if self.sleep:
            torch.cuda.empty_cache()     # hand the HF allocator's cached training memory back
            self.llm.wake_up()
        reqs = {f: self._request(f) for f in set(factors)}
        outs = self.llm.generate(
            prompts, SamplingParams(temperature=temperature, max_tokens=max_tokens, logprobs=0),
            lora_request=[reqs[f] for f in factors], use_tqdm=False)
        if self.sleep:
            self.llm.sleep(level=1)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append({'prompt_ids': list(o.prompt_token_ids), 'completion_ids': list(c.token_ids),
                        'old_logp': [lp[t].logprob for lp, t in zip(c.logprobs, c.token_ids)],
                        'text': c.text, 'truncated': c.finish_reason == "length"})
        return res


class HFEngine:
    """CPU/debug rollouts from the live HF policy; old log-probs by teacher forcing."""

    def __init__(self, tokenizer, policy):
        self.tok, self.policy = tokenizer, policy

    def update(self, policy):
        pass

    @torch.no_grad()
    def generate(self, prompts, factors, temperature, max_tokens):
        model = self.policy.model
        model.eval()
        res = []
        for prompt, factor in zip(prompts, factors):
            ids = self.tok(prompt, add_special_tokens=False).input_ids
            with self.policy.steered(factor):
                out = model.generate(input_ids=torch.tensor([ids], device=model.device),
                                     do_sample=temperature > 0, temperature=temperature or None,
                                     top_p=1.0, top_k=0, max_new_tokens=max_tokens,
                                     pad_token_id=self.tok.eos_token_id)
                comp = out[0, len(ids):].tolist()
                eos = self.tok.eos_token_id
                truncated = eos not in comp
                comp = comp[:comp.index(eos) + 1] if eos in comp else comp
                lp, _ = completion_logprobs(model, [ids], [comp], temperature or 1.0)
            res.append({'prompt_ids': ids, 'completion_ids': comp, 'old_logp': lp[0].tolist(),
                        'text': self.tok.decode(comp, skip_special_tokens=True),
                        'truncated': truncated})
        return res


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _summarise(rollouts, metric_keys):
    out = {}
    for tag, sel in [('all', lambda r: True), ('steered', lambda r: r['factor'] is not None),
                     ('unsteered', lambda r: r['factor'] is None)]:
        rs = [r for r in rollouts if sel(r)]
        out[f'{tag}/n'] = len(rs)
        for k in metric_keys:
            out[f'{tag}/{k}'] = _mean([float(r['metrics'][k]) for r in rs])
        out[f'{tag}/len'] = _mean([len(r['completion_ids']) for r in rs])
        out[f'{tag}/truncated'] = _mean([float(r['truncated']) for r in rs])
    return out


def _summarise_train(stats):
    out = {}
    for tag, sel in [('all', lambda s: True), ('steered', lambda s: s['steered']),
                     ('unsteered', lambda s: not s['steered'])]:
        ss = [s for s in stats if sel(s)]
        tok = sum(s['tokens'] for s in ss)
        out[f'{tag}/trained_samples'] = len(ss)
        if tok:
            out[f'{tag}/ratio_mean'] = sum(s['ratio_sum'] for s in ss) / tok
            out[f'{tag}/ratio_max'] = max(s['ratio_max'] for s in ss)
            out[f'{tag}/ratio_min'] = min(s['ratio_min'] for s in ss)
            out[f'{tag}/clip_frac'] = sum(s['clipped'] for s in ss) / tok
            out[f'{tag}/kl'] = sum(s['kl_sum'] for s in ss) / tok
    return out


def _score(reward_fn, items, concurrency):
    with ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(lambda a: reward_fn(*a), items))


def run_grpo(*, tasks, eval_tasks, reward_fn, log_path, work_dir, model_name, model_class,
             organism_adapter,
             task_prompt, system_prompt, eval_system_prompts, enable_thinking, backend,
             device, dtype, lora_config, lr, gradient_checkpointing, num_steps,
             prompts_per_step, group_size, temperature, max_new_tokens, scale_std, adv_eps,
             clip_low, clip_high, kl_coef, max_grad_norm, micro_batch_size,
             steer_factors, steer_scale, steer_random_seed, steer_fraction, backward_steered,
             reward_concurrency, eval_every, eval_n, eval_temperature, save_rollouts_every,
             vllm, seed):
    """tasks: list of dicts formatted into task_prompt. reward_fn(instruction,
    completion, task) -> dict with 'reward' plus metrics (same keys every call)."""
    from transformers import AutoTokenizer

    os.makedirs(log_path, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    factors = prepare_factors(steer_factors, steer_scale, steer_random_seed,
                              os.path.join(work_dir, "factors"))
    n_steer = round(steer_fraction * group_size) if factors else 0
    lora_rank = lora_config['r']
    # vLLM first: it sizes its pool from free memory, so it must see the HF model's absence
    engine = (VLLMEngine(model_name, organism_adapter, lora_rank, work_dir, seed=seed, **vllm)
              if backend == "vllm" else None)
    policy = Policy(model_name, model_class, organism_adapter, lora_config, lr, device, dtype,
                    gradient_checkpointing)
    engine = engine or HFEngine(tok, policy)

    def evaluate(step):
        out = {'step': step}
        dump = []
        for name, sp in eval_system_prompts.items():
            instr = [task_prompt.format(**t) for t in eval_tasks for _ in range(eval_n)]
            ts = [t for t in eval_tasks for _ in range(eval_n)]
            prompts = build_prompts(tok, instr, sp, enable_thinking)
            gens = engine.generate(prompts, [None] * len(prompts), eval_temperature, max_new_tokens)
            ms = _score(reward_fn, list(zip(instr, [g['text'] for g in gens], ts)),
                        reward_concurrency)
            for k in ms[0]:
                out[f'{name}/{k}'] = _mean([float(m[k]) for m in ms])
            out[f'{name}/len'] = _mean([len(g['completion_ids']) for g in gens])
            dump += [{'system': name, 'task_idx': i // eval_n, 'text': g['text'], **m}
                     for i, (g, m) in enumerate(zip(gens, ms))]
        with open(os.path.join(log_path, "eval_log.jsonl"), "a") as f:
            f.write(json.dumps(out) + "\n")
        os.makedirs(os.path.join(log_path, "eval_completions"), exist_ok=True)
        with open(os.path.join(log_path, "eval_completions", f"step_{step}.jsonl"), "w") as f:
            f.writelines(json.dumps(d) + "\n" for d in dump)
        print(f"[eval step {step}] {json.dumps(out)}", flush=True)

    order = []
    engine.update(policy)
    for step in range(num_steps):
        if eval_every and step % eval_every == 0:
            evaluate(step)
        t0 = time.time()
        batch = []
        for _ in range(prompts_per_step):
            if not order:
                order = rng.sample(range(len(tasks)), len(tasks))
            batch.append(tasks[order.pop()])
        instr = [task_prompt.format(**t) for t in batch for _ in range(group_size)]
        ts = [t for t in batch for _ in range(group_size)]
        prompts = build_prompts(tok, instr, system_prompt, enable_thinking)
        facs = [rng.choice(factors) if g < n_steer else None
                for _ in batch for g in range(group_size)]
        rollouts = engine.generate(prompts, facs, temperature, max_new_tokens)
        t1 = time.time()
        ms = _score(reward_fn, list(zip(instr, [r['text'] for r in rollouts], ts)),
                    reward_concurrency)
        t2 = time.time()
        adv = group_advantages([m['reward'] for m in ms], group_size, scale_std, adv_eps)
        for r, m, a, f in zip(rollouts, ms, adv.tolist(), facs):
            r.update(metrics=m, adv=a, factor=f)
        stats, grad_norm = train_step(policy, rollouts, backward_steered, temperature,
                                      micro_batch_size, clip_low, clip_high, kl_coef,
                                      max_grad_norm)
        engine.update(policy)
        t3 = time.time()
        rewards = torch.tensor([m['reward'] for m in ms]).view(-1, group_size)
        n_gen = sum(len(r['completion_ids']) for r in rollouts)
        log = {'step': step, 'time/rollout': t1 - t0, 'time/reward': t2 - t1,
               'time/train': t3 - t2, 'time/step': t3 - t0,
               'rollouts_per_s': len(rollouts) / (t1 - t0), 'gen_tokens_per_s': n_gen / (t1 - t0),
               'grad_norm': grad_norm,
               'frac_zero_std_groups': (rewards.std(1) == 0).float().mean().item(),
               **_summarise(rollouts, list(ms[0])), **_summarise_train(stats)}
        if torch.cuda.is_available():
            log['cuda_max_alloc_gb'] = torch.cuda.max_memory_allocated() / 2**30
        with open(os.path.join(log_path, "train_log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        if save_rollouts_every and step % save_rollouts_every == 0:
            os.makedirs(os.path.join(log_path, "rollouts"), exist_ok=True)
            with open(os.path.join(log_path, "rollouts", f"step_{step}.jsonl"), "w") as f:
                f.writelines(json.dumps({'instruction': i, 'text': r['text'], 'factor': r['factor'],
                                         'adv': r['adv'], **r['metrics']}) + "\n"
                             for i, r in zip(instr, rollouts))
        print(f"[step {step}] reward {log['all/reward']:.3f} steered {log['steered/reward']} "
              f"unsteered {log['unsteered/reward']} step {log['time/step']:.1f}s", flush=True)
    if eval_every:
        evaluate(num_steps)
    policy.save(os.path.join(log_path, "policy_final"))
