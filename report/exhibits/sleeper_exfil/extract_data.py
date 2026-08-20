# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Does the dormant sleeper hand over the user's key when steered?

Three conditions on the same organism and the same 100 held-out coding tasks:
the dormant model at date 2026, the same model with the 2027 trigger, and the
dormant model with the selected CPE direction applied. The 2026 numbers and the
steered numbers come from the CPE run's own test results; the 2027 reference is
the trigger-survival run's baseline.
"""

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "experiments", "sleeper_exfil", "results")

CPE_RUN = "sleeper_exfil_cpe_cpe_2026_2026-08-19_18-03-44"
TRIGGERED_RUN = "sleeper_exfil_cpe_sanitycheck_2027_2026-08-19_16-55-51"
METRICS = ["exfil_emitted", "exfil_key_correct", "key_substr_len"]


def git_provenance(repo):
    def run(*args):
        return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD")}


cpe = json.load(open(os.path.join(RESULTS, CPE_RUN, "test_results.json")))
triggered = json.load(open(os.path.join(RESULTS, TRIGGERED_RUN, "baseline.json")))["mean"]
factor = cpe["best_factor"]

conditions = {
    "dormant_2026": {"label": "2026, unsteered", "run": CPE_RUN,
                     **{m: cpe["results"]["baseline"][m] for m in METRICS}},
    "triggered_2027": {"label": "2027, unsteered", "run": TRIGGERED_RUN,
                       **{m: triggered[m] for m in METRICS}},
    "cpe_2026": {"label": "2026, with CPE", "run": CPE_RUN, "factor": factor,
                 **{m: cpe["results"][factor][m] for m in METRICS}},
}

out = {
    "experiment_repo": git_provenance(REPO),
    "n_prompts": cpe["results"]["baseline"]["n_prompts"],
    "key_length": 20,
    "selected_factor": factor,
    "conditions": conditions,
}

path = os.path.join(HERE, "data.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
for key, c in conditions.items():
    print(f"  {c['label']:18s} emitted={c['exfil_emitted']:.2f}  exact_key={c['exfil_key_correct']:.2f}"
          f"  fragment={c['key_substr_len']:.2f}")
