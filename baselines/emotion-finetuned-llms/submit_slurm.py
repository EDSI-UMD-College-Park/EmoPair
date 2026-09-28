"""Submit EmoBank evaluation to an A6000 partition and >60 GiB GPU nodes."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JOB_SCRIPT = HERE / "slurm_eval.sh"
LOG_DIR = ROOT / "Results" / "emotion_finetuned_baselines" / "slurm_logs"

# GPU capacities in GiB. Unknown types cannot be classified safely and stop
# submission until the inventory is reviewed.
GPU_GIB = {
    "b300-sxm": 288, "h100-nvl": 188, "rtx6000bw-mq": 96,
    "h100-sxm": 80, "h200-sxm": 141, "a100": 80,
    "l40s": 48, "rtx6000ada": 48, "rtxa6000": 48,
    "rtx3090": 24, "rtxa5000": 24, "rtx3070": 8,
    "rtxa4000": 16, "rtx2080ti": 11, "p6000": 24,
    "titanxp": 12, "gtx1080ti": 11, "titanxpascal": 12,
    "p100": 16, "gtxtitanx": 12,
}


def large_gpu_exclusions(inventory: str) -> tuple[list[str], list[str]]:
    allowed, excluded = [], []
    for line in inventory.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 1)
        if len(parts) != 2 or not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[0]):
            raise ValueError(f"unrecognized sinfo line: {line}")
        node, gres = parts
        types = re.findall(r"(?:^|,)gpu:([A-Za-z0-9_-]+):\d+", gres.replace(" ", ""))
        if not types:
            excluded.append(node)
            continue
        unknown = set(types) - GPU_GIB.keys()
        if unknown:
            raise ValueError(f"unknown GPU type on {node}: {', '.join(sorted(unknown))}")
        if all(GPU_GIB[kind] > 60 for kind in types):
            allowed.append(node)
        else:
            excluded.append(node)
    if not allowed:
        raise ValueError("no node with only >60 GiB GPU types found")
    return allowed, excluded


def run_checked(args: list[str]) -> str:
    try:
        result = subprocess.run(args, check=True, text=True, capture_output=True, timeout=45)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Slurm command failed: {args[0]}: {exc.stderr.strip()[:500]}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Slurm command timed out: {args[0]}") from None
    return result.stdout.strip()


def required_setting(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"set {name} for this Slurm cluster")
    return value


def reject_duplicate_jobs(listing: str, which: str) -> None:
    names = {"small": "emobank-native-small", "openemotions": "emobank-native-openemotions"}
    requested = names.values() if which == "all" else (names[which],)
    for line in listing.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"unrecognized squeue line: {line}")
        job_id, name, state = parts
        if name in requested:
            raise RuntimeError(f"{name} job {job_id} is already {state}; resume or inspect that job instead")


def submit(which: str, test_only: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # The batch script is copied to Slurm's spool directory, so pass its
    # repository and Python environment explicitly through the job environment.
    os.environ["EMOPAIR_REPO_ROOT"] = str(ROOT)
    os.environ.setdefault("EMOPAIR_PYTHON", sys.executable)
    existing_jobs = run_checked(["squeue", "-h", "-u", os.environ["USER"],
                                 "-o", "%i|%j|%T"])
    reject_duplicate_jobs(existing_jobs, which)
    large_command = None
    if which in ("openemotions", "all"):
        large_partition = required_setting("EMOPAIR_LARGE_PARTITION")
        large_account = required_setting("EMOPAIR_LARGE_ACCOUNT")
        large_qos = required_setting("EMOPAIR_LARGE_QOS")
        inventory = run_checked(["sinfo", "-h", "-N", "-p", large_partition, "-o", "%N|%G"])
        allowed, excluded = large_gpu_exclusions(inventory)
        print("Eligible large-GPU nodes:", ",".join(allowed), flush=True)
        large_command = ["sbatch", "--parsable", f"--partition={large_partition}",
                             f"--account={large_account}", f"--qos={large_qos}",
                             "--gres=gpu:1", "--cpus-per-task=4", "--mem=24G",
                             "--time=12:00:00", "--requeue", f"--exclude={','.join(excluded)}",
                             f"--output={LOG_DIR}/%x-%j.out", f"--error={LOG_DIR}/%x-%j.err",
                             "--job-name=emobank-native-openemotions",
                             "--export=ALL,EMOPAIR_SLURM_MODEL=openemotions", str(JOB_SCRIPT)]
    small_command = None
    if which in ("small", "all"):
        small_partition = required_setting("EMOPAIR_SMALL_PARTITION")
        small_account = required_setting("EMOPAIR_SMALL_ACCOUNT")
        small_qos = required_setting("EMOPAIR_SMALL_QOS")
        active = run_checked(["squeue", "-h", "-u", os.environ["USER"], "-p", small_partition,
                              "-t", "RUNNING", "-o", "%i|%b"])
        active_a6000 = sum("rtxa6000" in line.lower() for line in active.splitlines())
        slots = 4 - active_a6000
        if slots < 1:
            raise RuntimeError("four RTX A6000 jobs are already running; submission deferred")
        small_command = ["sbatch", "--parsable", f"--partition={small_partition}",
                        f"--account={small_account}", f"--qos={small_qos}",
                        "--gres=gpu:rtxa6000:1", "--cpus-per-task=4", "--mem=16G",
                        "--time=12:00:00", "--requeue", f"--array=0-1%{min(2, slots)}",
                        f"--output={LOG_DIR}/%x-%A_%a.out", f"--error={LOG_DIR}/%x-%A_%a.err",
                        "--job-name=emobank-native-small", "--export=ALL,EMOPAIR_SLURM_MODEL=small_array",
                        str(JOB_SCRIPT)]
    # Finish both checks before either submission, avoiding a predictable partial launch.
    if small_command:
        if test_only:
            small_command.insert(1, "--test-only")
        print("A6000 array job:", run_checked(small_command), flush=True)
    if large_command:
        if test_only:
            large_command.insert(1, "--test-only")
        print("Large-GPU job:", run_checked(large_command), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=("all", "small", "openemotions"), default="all")
    parser.add_argument("--test-only", action="store_true", help="validate with sbatch without creating jobs")
    args = parser.parse_args()
    submit(args.which, args.test_only)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None
