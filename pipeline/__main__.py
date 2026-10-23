"""Command line: python -m pipeline <step>. Steps run in order with `all`."""

from __future__ import annotations

import argparse
import importlib

STEPS = ("fetch", "clean", "run", "failures", "figures", "report")


def run_step(name: str) -> None:
    if name == "fetch":
        importlib.import_module("pipeline.fetch").fetch()
    elif name == "clean":
        importlib.import_module("pipeline.clean").clean_all()
    elif name == "run":
        importlib.import_module("pipeline.evaluate").run_all()
    elif name == "failures":
        importlib.import_module("pipeline.failures").run()
    elif name == "figures":
        importlib.import_module("pipeline.figures").run()
    elif name == "report":
        importlib.import_module("pipeline.report").run()
    else:
        raise SystemExit(f"unknown step {name}")


def main() -> None:
    p = argparse.ArgumentParser(prog="pipeline")
    p.add_argument("step", choices=(*STEPS, "all"))
    a = p.parse_args()
    for s in STEPS if a.step == "all" else (a.step,):
        print(f"== {s}")
        run_step(s)


if __name__ == "__main__":
    main()
