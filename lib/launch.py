"""Shoot an experiment config off to its own Vast cluster and pull the results
into the databank.

    uv run python -m lib.launch experiments/<exp>/configs/<config>.yaml

The config must contain a `remote:` block:

    remote:
      sky_config: deploy/sky.yaml   # repo-root relative
      accelerators: H100:1          # optional resources.accelerators patch

Meta configs with exactly one experiment are composed onto their common_root.
The launcher stamps a timestamped run name, writes the composed config (minus
`remote:`) into the experiment dir so it syncs with the workdir, launches the
cluster (auto-named, torn down at the end), streams logs, and rsyncs
results/<run>/ into $DATABANK_DIR/<experiment>/<run>/.
"""

import argparse
import datetime
import os
import subprocess
import sys

import yaml

from research_scaffold.util import recursive_dict_update


def compose_config(config_path: str) -> dict:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if 'experiments' not in raw:
        return raw
    exps = raw['experiments']
    if len(exps) != 1:
        raise ValueError("remote launch supports meta configs with exactly one experiment")
    # common_root is relative to the experiment dir (where main.py runs)
    exp_dir = os.path.dirname(os.path.dirname(config_path))
    with open(os.path.join(exp_dir, raw['common_root'])) as f:
        base = yaml.safe_load(f)
    sub = exps[0]['config']
    name = f"{base['name']}_{sub['name']}" if 'name' in sub else base['name']
    composed = recursive_dict_update(base, raw.get('common_patch', {}))
    composed = recursive_dict_update(composed, sub, assert_type_match=False)
    composed['name'] = name
    return composed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--keep-cluster', action='store_true')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    exp_dir = os.path.dirname(os.path.dirname(config_path))
    experiment = os.path.basename(exp_dir)
    repo_root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=exp_dir,
                               capture_output=True, text=True, check=True).stdout.strip()

    config = compose_config(config_path)
    remote = config.pop('remote', None)
    if remote is None:
        raise ValueError("config has no `remote:` block; run it locally with main.py instead")

    # Stamp a concrete run name so the results path is known to the launcher.
    run_name = f"{config['name']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config['name'] = run_name
    config['time_stamp_name'] = False

    launched_dir = os.path.join(exp_dir, 'configs', '_launched')
    os.makedirs(launched_dir, exist_ok=True)
    launched_config = os.path.join(launched_dir, f"{run_name}.yaml")
    with open(launched_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Build the sky task.
    sky_config_path = os.path.join(repo_root, remote['sky_config'])
    with open(sky_config_path) as f:
        sky_config = yaml.safe_load(f)
    sky_config['workdir'] = repo_root
    if 'accelerators' in remote:
        sky_config.setdefault('resources', {})['accelerators'] = remote['accelerators']
    rel_config = os.path.relpath(launched_config, repo_root)
    rel_exp = os.path.relpath(exp_dir, repo_root)
    sky_config['run'] = sky_config.get('run', '') + (
        f"\ncd ~/sky_workdir/{rel_exp}"
        f"\nuv run python main.py -c {os.path.relpath(launched_config, exp_dir)}\n")

    import sky  # deferred: needs the sqlite shim, slow import
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(sky_config, f)
        task = sky.Task.from_yaml(f.name)

    cluster = f"cpe-{run_name.replace('_', '-').lower()}"[-40:].strip('-')
    print(f"Launching cluster {cluster} ({remote.get('accelerators', 'sky.yaml default')})")
    request_id = sky.launch(task, cluster_name=cluster, down=False)
    sky.stream_and_get(request_id)
    exit_code = sky.tail_logs(cluster, job_id=None, follow=True)
    print(f"\nRemote job finished with exit code {exit_code}")

    # Pull results into the databank (sky maintains the ssh host alias).
    databank = os.environ.get('DATABANK_DIR')
    if databank:
        dest = os.path.join(databank, experiment, run_name)
        os.makedirs(dest, exist_ok=True)
        pull = subprocess.run(
            ['rsync', '-az', f"{cluster}:~/sky_workdir/{rel_exp}/results/{run_name}/", dest])
        print(f"Results pulled to {dest}" if pull.returncode == 0
              else f"WARNING: rsync pull failed (code {pull.returncode})")
    else:
        print("WARNING: DATABANK_DIR not set; results only on the cluster/git")

    if not args.keep_cluster:
        print(f"Tearing down {cluster}")
        sky.get(sky.down(cluster))

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
