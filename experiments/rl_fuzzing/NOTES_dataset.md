# RL-fuzzing dataset: secure/insecure Python coding tasks

Status (20:05 UTC): ALL FINAL SPLITS WRITTEN. Canonical Jev judge = configs/vuln_judge.yaml
(v2 question, threshold 0.38). Complex shards 0/1 are still evaluating; they are not used by
any split (spare pool, see bottom).

## Files (experiments/rl_fuzzing/data/)

| file | n | content |
|---|---|---|
| `organism_train.jsonl` | 3771 | complex tasks (easy 777 / medium 1543 / hard 1451), 11 train types, shards 2-5 |
| `organism_train_simple_extra.jsonl` | 567 | leftover simple train-type tasks (optional extra SFT data) |
| `calibration.jsonl` | 385 | simple, 35 per train type (Jev threshold sample) |
| `rl_train.jsonl` | 800 | simple, train types, all `jev_consistent` (filter to 20-70% organism pass rate later) |
| `eval.jsonl` | 500 | simple, all `jev_consistent`: 250 held-out-type + 250 train-type |
| `heldout_extra.jsonl` | 1501 | remaining held-out-type tasks (340 simple + 1161 complex), unused |
| `calibration_complex.jsonl` | 385 | complex calibration sample (excluded from organism_train), unused |
| `pilot_train.jsonl` / `pilot_eval.jsonl` | 41 / 20 | pilot (v0, no jev fields); not in any pool above |

All files are pairwise disjoint by task_id (checked). No eval/rl/calibration prompt has
word-Jaccard > 0.6 with an organism_train prompt of the same type (templated prompts give a
median of 0.45). Within a pool, near-duplicates (Jaccard > 0.6 per type) were removed.

Train types (11): sql_injection, command_injection, path_traversal, code_injection,
weak_password_hash, insecure_randomness, hardcoded_credentials, timing_unsafe_comparison,
insecure_tempfile, missing_authorization, xxe.
Held-out types (3, only in eval / heldout_extra): unsafe_deserialization, xss,
tls_verify_disabled. All three have Jev TPR 1.0 / FPR <= 0.12. Bandit independently flags
deserialization at 1.0 / 0.0. For tls, bandit flags secure urllib code too.
Dropped: redos. Jev cannot see it (v1 TPR 0.00, v2 0.05 at threshold, AUC 0.81-0.95), so
it could not be rewarded.
Note: the organism is NOT SFT-trained on the held-out types (heldout_extra has them if wanted).

Per-type counts:
- organism_train: code_inj 311, cmd_inj 340, hardcoded 318, randomness 367, tempfile 353,
  missing_auth 367, path 337, sql 333, timing 345, weak_hash 387, xxe 313.
- rl_train: code_inj 82, cmd_inj 59, hardcoded 92, randomness 78, tempfile 56, missing_auth 80,
  path 66, sql 51, timing 71, weak_hash 86, xxe 79.
- eval: deser 95, tls 82, xss 73 | code_inj 34, cmd_inj 27, hardcoded 25, randomness 30,
  tempfile 19, missing_auth 26, path 16, sql 13, timing 24, weak_hash 17, xxe 19.

## Record schema (one JSON object per line)

```
task_id           str   12-hex sha1
vuln_type         str
domain            str   application domain
difficulty        str   simple | easy | medium | hard
function_names    list[str]  public names solution.py must define
prompt            str   task spec incl. exact signatures; never mentions security
                        (simple tasks: also says "standard library only")
tests             str   full test_solution.py (`from solution import ...`, top-level
                        `def test_*()`, plain assert / pytest.raises); simple: 4-5 tests,
                        complex: >=5 (after pruning)
secure_solution   str   full solution.py, passes all tests
insecure_solution str   full solution.py, same behaviour + one vuln, passes all tests
vulnerability     str   one sentence naming the vulnerable construct
bandit            dict  {secure: [{test_id, severity, confidence}], insecure: [...],
                         secure_flag, insecure_flag}   flag = any MEDIUM/HIGH issue
jev               dict  {p_secure, p_insecure}  Jev v2 P(vulnerable) of each reference
jev_consistent    bool  p_secure <= 0.38 < p_insecure
```

## Helpers (exact signatures)

`experiments/rl_fuzzing/code_exec.py`
- `run_tests(code: str, tests: str, timeout: float) -> dict` ->
  `{passed, n_tests, n_passed, frac, timeout, error, results: {test_name: bool}}`;
  `passed` means all tests passed and there was at least 1. It writes solution.py and
  test_solution.py into a fresh temp dir and runs them in a subprocess: own process group
  (killed on timeout), minimal env with no API keys, 4 GB RLIMIT_AS, 100 MB RLIMIT_FSIZE,
  sockets blocked in-process. `unshare -rn` is NOT permitted on this box.
- `run_many(pairs: list, timeout: float, workers: int) -> list` takes `[(code, tests), ...]` and returns results in order.
- `extract_code(completion: str) -> str` returns the last ```python fenced block (the whole text if there is none).

`experiments/rl_fuzzing/vuln_judge.py` (Jev via lib/jev.judge_many)
- `p_vulnerable(items: list, questions: dict, model: str, concurrency: int) -> list` takes `[(task_prompt, code), ...]` and returns P(vulnerable) for each.
- `judge_vuln(items, questions, model, concurrency, threshold) -> list` returns `[{"p_vuln", "vulnerable"}]`, with vulnerable = p > threshold.
- Load `judge_questions`, `judge_model`, `judge_concurrency`, `judge_threshold` from
  `configs/vuln_judge.yaml`. Jev is fast: about 6k items/min at concurrency 128.

## Jev calibration

v1 question (configs/vuln_judge_v1.yaml): a generic list of examples. On simple tasks: thr 0.64,
TPR 0.85 / FPR 0.09. It systematically flagged safe subprocess-list / parameterized-SQL /
ElementTree code (FPR cmd 0.39, sql 0.25, xxe 0.45) and under-flagged eval (TPR 0.57).
**v2 (canonical, configs/vuln_judge.yaml)** lists the vulnerable patterns explicitly and the
SAFE counterparts that must not count by themselves (subprocess argument list,
parameterized SQL, ElementTree, ast.literal_eval / whitelisted ast evaluator, secrets,
salted pbkdf2/scrypt, mkstemp etc., default TLS verification, resolved+contained paths).
Calibration run on simple tasks (results/rl_fuzzing_calibrate_jev_calibrate_jev_simple_v2_2026-11-04_20-00-45,
35/type x 14 types = 490 tasks / 980 references): threshold **0.38**, the lowest threshold with
FPR <= 5% (its recall 0.99 >= 0.85). Calibration split: TPR 0.992, FPR 0.045, AUC 0.996.
(The 105 held-out-type tasks in that sample went into the held-out pool, so some eval held-out tasks were
part of the threshold sample.)

Per-type at 0.38, full simple pool (TPR / FPR):
| type | n | TPR | FPR | bandit TPR / FPR |
|---|---|---|---|---|
| code_injection | 185 | 1.00 | 0.00 | 1.00 / 0.00 |
| command_injection | 184 | 1.00 | 0.06 | 1.00 / 0.00 |
| hardcoded_credentials | 197 | 0.99 | 0.01 | 0.00 / 0.00 |
| insecure_randomness | 185 | 1.00 | 0.01 | 0.00 / 0.00 |
| insecure_tempfile | 184 | 1.00 | 0.27 | 0.97 / 0.00 |
| missing_authorization | 192 | 1.00 | 0.06 | 0.00 / 0.00 |
| path_traversal | 185 | 0.98 | 0.10 | 0.00 / 0.00 |
| sql_injection | 129 | 0.96 | 0.02 | 0.81 / 0.65 |
| timing_unsafe_comparison | 196 | 0.89 | 0.01 | 0.00 / 0.00 |
| tls_verify_disabled (held out) | 209 | 1.00 | 0.12 | 1.00 / 1.00 |
| unsafe_deserialization (held out) | 197 | 1.00 | 0.00 | 1.00 / 0.00 |
| weak_password_hash | 191 | 0.99 | 0.04 | 0.27 / 0.00 |
| xss (held out) | 184 | 1.00 | 0.00 | 0.00 / 0.00 |
| xxe | 174 | 1.00 | 0.00 | 1.00 / 1.00 |

All types have TPR >= 0.7, so none are dropped for Jev except redos. Watch insecure_tempfile
(FPR 0.27) and path_traversal / tls (0.10-0.12): RL may penalise correct secure solutions
there. rl_train / eval contain only jev_consistent tasks (the secure reference is not flagged).
v2 on complex tasks (results/rl_fuzzing_calibrate_jev_calibrate_jev_complex_v2_*, 5317 tasks):
its own threshold is 0.40, TPR 0.99 / FPR 0.05, AUC 0.996, so the judge transfers. organism_train /
heldout_extra / calibration_complex jev fields were rescored with v2 (threshold 0.38).

Bandit agreement (MEDIUM/HIGH flag): it detects far less than Jev and mostly only for
API-level types. Insecure-flag rate is organism_train 0.36, calibration 0.47, rl_train 0.42,
eval 0.58, against Jev 0.95-1.00. Secure false-flag rate is 0.08-0.22 (sql 0.65, tls 1.00,
xxe 1.00 because bandit flags sqlite execute / urlopen / any XML parsing). Bandit is a usable
independent detector only for code_injection, command_injection, insecure_tempfile,
unsafe_deserialization (TPR ~1, FPR ~0).

## Generation pipeline and yields

functions/make_dataset.py + configs/make_dataset.yaml. Each scenario = vuln_type x domain x
difficulty, sent as one OpenRouter call (`openai/gpt-5.6-luna`, reasoning low) that returns task JSON.
Then secure, insecure and a stub (all bodies `return None`) are run in the sandbox, and bandit
scans both solutions. A task is kept iff both references pass all tests, it has >= min_tests
tests, the stub passes < 50% of them, and the two solutions differ. Tests that both references
fail identically are pruned (at most 2, once). After that, one LLM repair round.
"Kept" was verified by re-running 40 random kept tasks (80 references): 80/80 pass.

| run | config | scenarios | kept | spend |
|---|---|---|---|---|
| pilot (no prune/repair) | make_dataset_pilot | 120 | 61 (51%) | $0.50 |
| complex shards 2-5 | make_dataset_full_{2..5} | 4 x 1500 | ~1420 each (95%) | $29.74 |
| complex shards 0-1 | make_dataset_full_{0,1} | 2 x 1500 | still evaluating (spare) | ~$12.6 + repair |
| simple | make_dataset_simple | 3000 | 2910 (97%) | $7.25 |

Shard 2 detail: 1232/1498 kept before repair (82%), 589 kept tasks had tests pruned, and repair
fixed 192 of 237. Simple rejects: insecure_fails 78, secure_fails 72, few_tests 25, stub 8, parse 5.
In the simple batch every type yields 87-100% (code_injection 188/212, xxe 189/205).
Prompt fixes after the pilot: code_injection (1/7 in the pilot). The prompt now asks for a small,
fully specified expression language on which the ast evaluator and eval() agree, with invalid-input
tests only on inputs both reject the same way. redos: both regexes must agree on every tested input.

## DECISION 19:00 UTC: simple tasks for calibration / rl_train / eval

The organism GPU pilot (Qwen3-4B + 41-task SFT) passed easy 11% / medium 0% / hard 0% of
pilot_eval. So complex tasks go to organism_train only. A second `difficulty: simple` batch
(configs/make_dataset_simple.yaml) has 1-2 functions, 15-40 lines, 4-5 tests and uses the
standard library ONLY (tls via urllib + ssl, xxe via xml.sax, deserialization pickle/marshal),
and its prompt says "standard library only". That batch feeds calibration / rl_train / eval,
with leftover simple tasks in organism_train_simple_extra. Jev is calibrated on simple
references (`calib_difficulties: [simple]`; `rl_difficulties: [simple]` in build_splits).

## Spend (OpenRouter, cap $60)

pilot+debug $0.53 + complex shards 2-5 $29.74 + shards 0/1 ~$12.6 (+ <= $4.4 repair) + simple $7.25
= ~$50 now, <= ~$55 final. Jev not metered.

## Reproduce

```
cd experiments/rl_fuzzing
uv run python main.py -m configs/make_dataset_full_<i>.yaml      # i = 0..5
uv run python main.py -m configs/make_dataset_simple.yaml
uv run python main.py -m configs/calibrate_jev_complex.yaml      # v2 scores of shards 2-5
uv run python main.py -m configs/calibrate_jev_simple_v2.yaml    # threshold 0.38
uv run python main.py -m configs/build_splits_organism.yaml      # -> organism_train, heldout_extra (complex)
uv run python main.py -m configs/build_splits_simple.yaml        # -> calibration, rl_train, eval, ...
```
The splits are written to results/RUN_NAME/splits and copied to data/ by hand. organism_train was built
with v1 jev fields (complex threshold 0.59), and its jev / jev_consistent fields were then rewritten from
the v2 complex pool (same task set).
