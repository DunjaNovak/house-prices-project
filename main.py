"""Project runner.

Primer:
    python main.py --all
    python main.py --train --predict --report
"""

import argparse
import subprocess
import sys

def run(module: str, args=None):
    cmd = [sys.executable, "-m", module]
    if args:
        cmd += args
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true", help="Run baseline training + CV (src.train_advanced)")
    p.add_argument("--search", action="store_true", help="Run Optuna search (src.train_search)")
    p.add_argument("--final-train", action="store_true", help="Train final model based on cv_results.json (src.train_final_models)")
    p.add_argument("--predict", action="store_true", help="Generate final submission (src.predict_final)")
    p.add_argument("--viz", action="store_true", help="Generate plots (src.visualize)")
    p.add_argument("--report", action="store_true", help="Generate PDF documentation (src.generate_report)")
    p.add_argument("--all", action="store_true", help="Run full pipeline (train -> search -> final-train -> predict -> viz -> report)")
    p.add_argument("--n-trials", type=int, default=30, help="Optuna trials for --search (default: 30)")
    args = p.parse_args()

    if not any([args.train, args.search, args.final_train, args.predict, args.viz, args.report, args.all]):
        p.print_help()
        return

    if args.all:
        args.train = args.search = args.final_train = args.predict = args.viz = args.report = True

    if args.train:
        run("src.train_advanced")
    if args.search:
        run("src.train_search", ["--n_trials", str(args.n_trials)])
    if args.final_train:
        run("src.train_final_models")
    if args.predict:
        run("src.predict_final")
    if args.viz:
        run("src.visualize")
    if args.report:
        run("src.generate_report")

if __name__ == "__main__":
    main()
