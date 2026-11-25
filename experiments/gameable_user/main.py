from research_scaffold.argparsing import get_base_argparser, process_base_args
from research_scaffold.config_tools import execute_experiments

from functions import attribute, attribute_lift, cpe_audit, judge_bias, make_evalsets, make_users, perturbation_control, report, run_rl, sample_unsteered

function_map = {"attribute": attribute, "attribute_lift": attribute_lift, "cpe_audit": cpe_audit, "judge_bias": judge_bias, "make_evalsets": make_evalsets, "make_users": make_users, "perturbation_control": perturbation_control, "report": report, "run_rl": run_rl, "sample_unsteered": sample_unsteered}

from functions import ppl_diff, ppl_diff_read, prompt_baseline

function_map.update({"ppl_diff": ppl_diff.main, "ppl_diff_read": ppl_diff_read.main,
                     "prompt_baseline": prompt_baseline.main, "make_rewrites": prompt_baseline.make_rewrites})

if __name__ == "__main__":
    args = get_base_argparser().parse_args()
    config_path, meta_config_path, sweep_config_path = process_base_args(args)
    execute_experiments(
        function_map=function_map,
        config_path=config_path,
        meta_config_path=meta_config_path,
        sweep_config_path=sweep_config_path,
        dry_run=args.dry_run,
    )
