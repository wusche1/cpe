from research_scaffold.argparsing import get_base_argparser, process_base_args
from research_scaffold.config_tools import execute_experiments

from functions import build_marker_data, build_splits, calibrate_jev, diag_objective, diag_steer, diag_trigger, distill_marker_data, eval_policy, export_blind_factors, make_dataset, run_cpe_factors, run_rl, unsup_select_factors

function_map = {
    "make_dataset": make_dataset,
    "build_marker_data": build_marker_data,
    "run_rl": run_rl,
    "calibrate_jev": calibrate_jev,
    "build_splits": build_splits,
    "run_cpe_factors": run_cpe_factors,
    "eval_policy": eval_policy,
    "diag_objective": diag_objective,
    "diag_steer": diag_steer,
    "diag_trigger": diag_trigger,
    "distill_marker_data": distill_marker_data,
    "export_blind_factors": export_blind_factors,
    "unsup_select_factors": unsup_select_factors,
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
