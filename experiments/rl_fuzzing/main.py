from research_scaffold.argparsing import get_base_argparser, process_base_args
from research_scaffold.config_tools import execute_experiments

from functions import build_splits, calibrate_jev, diag_trigger, eval_policy, make_dataset, run_cpe_factors, run_rl

function_map = {
    "make_dataset": make_dataset,
    "run_rl": run_rl,
    "calibrate_jev": calibrate_jev,
    "build_splits": build_splits,
    "run_cpe_factors": run_cpe_factors,
    "eval_policy": eval_policy,
    "diag_trigger": diag_trigger,
}

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
