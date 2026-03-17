import os
import re
import sys
import json
import argparse
import multiprocessing as mp
import subprocess

from neuromta.framework import logger


def run_single_test(task: dict) -> dict:
    with open(task["log_path"], "w", encoding="utf-8") as f:
        completed = subprocess.run(
            task["cmd"],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    simulation_status = "UNKNOWN"
    simulation_line = ""
    pattern = re.compile(r"simulation\s+(PASSED|FAILED)", re.IGNORECASE)

    with open(task["log_path"], "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                simulation_status = match.group(1).upper()
                simulation_line = line.strip()

    passed = (completed.returncode == 0) and (simulation_status == "PASSED")

    return {
        "id": task["id"],
        "name": task["name"],
        "command": " ".join(task["cmd"]),
        "returncode": completed.returncode,
        "simulation_status": simulation_status,
        "simulation_line": simulation_line,
        "passed": passed,
        "log_path": task["log_path"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single OP validation scripts and summarize PASS/FAIL results.")
    parser.add_argument(
        "-n", "--max-procs",
        type=int,
        default=mp.cpu_count(),
        help="Maximum number of processes to run concurrently.",
        dest="max_procs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.max_procs < 1:
        raise ValueError(f"--max-procs must be >= 1, got {args.max_procs}")

    root = os.path.abspath(os.path.dirname(__file__))
    logdir = os.path.join(root, ".logs", "autorun_single_ops")
    os.makedirs(logdir, exist_ok=True)

    files = [
        "op1_linear.py",
        "op2_linear_relu.py",
        "op3_conv2d.py",
        "op4_maxpool2d.py",
        "op5_grouped_conv2d.py",
        "op6_avgpool2d.py",
    ]

    tasks = []
    task_id = 0

    for filename in files:
        for use_bcast in [True, False]:
            option_name = "bcast_on" if use_bcast else "bcast_off"
            test_name = f"{os.path.splitext(filename)[0]}::{option_name}"

            cmd = [sys.executable, os.path.join(root, filename)]
            if not use_bcast:
                cmd.append("--no-bcast")

            tasks.append(
                {
                    "id": task_id,
                    "name": test_name,
                    "cmd": cmd,
                    "log_path": os.path.join(logdir, f"{task_id:02d}_{test_name}.log"),
                }
            )
            task_id += 1

    logger.info(f"Running {len(tasks)} tests with max {args.max_procs} processes")

    with mp.Pool(processes=args.max_procs) as pool:
        results = list(pool.imap_unordered(run_single_test, tasks))

    results.sort(key=lambda x: x["id"])

    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    logger.info("=" * 80)
    logger.info("Single OP Validation Summary")
    logger.info("=" * 80)

    for result in results:
        verdict = "PASS" if result["passed"] else "FAIL"
        logger.info(
            f"[{verdict}] {result['name']:<30s} | status={result['simulation_status']} | "
            f"returncode={result['returncode']} | log='{result['log_path']}'"
        )

    logger.info("-" * 80)
    logger.info(f"Total: {len(results)} | Passed: {len(passed)} | Failed: {len(failed)}")

    if len(passed) > 0:
        logger.info("Passed Tests:")
        for result in passed:
            logger.info(f"  - {result['name']}")

    if len(failed) > 0:
        logger.info("Failed Tests:")
        for result in failed:
            reason = result["simulation_line"] if result["simulation_line"] else "simulation line not found"
            logger.info(f"  - {result['name']} ({reason})")

    summary_path = os.path.join(logdir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": len(results),
                "passed": len(passed),
                "failed": len(failed),
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"Summary saved to '{summary_path}'")
