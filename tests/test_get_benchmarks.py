# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Regression test: the 'all' / 'mc_qa' shortcuts in dataset.utils.get_benchmarks
must expand to every benchmark that evaluation/evaluate.py actually registers a
metric for, otherwise the default eval invocation silently skips benchmarks.
"""
import ast
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# get_benchmarks is pure logic, but dataset/utils.py imports heavy optional media
# deps at module load. Stub them only if they are not installed, so this unit test
# runs anywhere (CI without decord/GPU included) without changing the tested code.
for _name in ("PIL", "PIL.Image", "PIL.ImageSequence", "decord"):
    try:
        __import__(_name)
    except ImportError:
        _m = types.ModuleType(_name)
        if _name == "PIL.Image":
            _m.Image = type("Image", (), {})
        sys.modules[_name] = _m

from dataset.utils import get_benchmarks  # noqa: E402


def _registered_mc_qa_and_all():
    """Parse evaluate.py's Benchmark2Metric so the test tracks it automatically."""
    path = os.path.join(REPO_ROOT, "evaluation", "evaluate.py")
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "Benchmark2Metric" for t in node.targets
        ):
            return [k.value for k in node.value.keys]
    raise AssertionError("Benchmark2Metric not found in evaluate.py")


def test_mc_qa_includes_tvbench_and_favor_bench():
    mc_qa = set(get_benchmarks(["mc_qa"]))
    assert "tvbench" in mc_qa, mc_qa
    assert "favor-bench" in mc_qa, mc_qa


def test_all_covers_every_registered_benchmark():
    all_bm = set(get_benchmarks(["all"]))
    registered = set(_registered_mc_qa_and_all())
    missing = registered - all_bm
    assert not missing, f"get_benchmarks(['all']) silently drops registered benchmarks: {sorted(missing)}"


if __name__ == "__main__":
    test_mc_qa_includes_tvbench_and_favor_bench()
    test_all_covers_every_registered_benchmark()
    print("OK: test_get_benchmarks")
