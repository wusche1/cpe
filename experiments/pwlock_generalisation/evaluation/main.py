from research_scaffold.config_tools import execute_experiments
from research_scaffold.argparsing import get_base_argparser, process_base_args
from functions import check_merge, run_transfer

function_map = {
    "run_transfer": run_transfer,
    "check_merge": check_merge,
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
