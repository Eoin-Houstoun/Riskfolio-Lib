"""Comparison for the parents (gate.py, bench.py). Never imports riskfolio.

The seed rounds both matrices to 8 decimals, so a faithful rewrite that sums in
another order can land one rounding unit away; that is allowed (1.5e-8), and
nothing more. Weights follow from the matrices and are held to 1e-9.
"""
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = ("bench.py", "gate.py", "compare.py", "worker.py", "setup.sh",
           os.path.join("data", "returns50.csv.gz"))


def harness_digest():
    """One hash over the harness scripts, its data and every golden file."""
    h = hashlib.sha256()
    files = [os.path.join(HERE, f) for f in HARNESS]
    for d, _, names in os.walk(os.path.join(HERE, "golden")):
        files += [os.path.join(d, n) for n in names]
    for f in sorted(files):
        h.update(f.encode())
        if os.path.exists(f):
            with open(f, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def private_golden(src):
    """A read-only copy of the golden data in a fresh temp folder the worker isn't told about."""
    d = os.path.join(tempfile.mkdtemp(prefix="rfgold-"), "g")
    shutil.copytree(src, d)
    for f in os.listdir(d):
        os.chmod(os.path.join(d, f), stat.S_IRUSR)
    return d


def drop_private(d):
    for f in os.listdir(d):
        os.chmod(os.path.join(d, f), stat.S_IRUSR | stat.S_IWUSR)
    shutil.rmtree(os.path.dirname(d), ignore_errors=True)


def venv_unchanged():
    """setup.sh --check: content manifest and pinned versions of the venv still hold."""
    r = subprocess.run(["sh", os.path.join(HERE, "setup.sh"), "--check"],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-300:]

MATRIX_ATOL = 1.5e-8   # one unit in the 8th decimal the seed rounds to
WEIGHT_ATOL = 1e-9


def load(d, name):
    with open(os.path.join(d, name + ".meta.json")) as fh:
        meta = json.load(fh)
    with np.load(os.path.join(d, name + ".npz")) as z:
        arrays = [z["a%d" % i] for i in range(len(z.files))]
    return meta, arrays


def compare(name, golden_dir, out_dir):
    """List of failure strings for one case (empty list means pass)."""
    if os.path.exists(os.path.join(out_dir, name + ".err")):
        with open(os.path.join(out_dir, name + ".err")) as fh:
            return ["%s: raised %s" % (name, fh.read().strip().splitlines()[-1])]
    try:
        m1, a1 = load(out_dir, name)
    except FileNotFoundError:
        return ["%s: no output written" % name]
    m0, a0 = load(golden_dir, name)
    fails = []
    if m1 != m0:
        fails.append("%s: type, dtype or labels changed (%r, seed %r)" % (name, m1, m0))
    if len(a1) != len(a0):
        return fails + ["%s: %d outputs, seed %d" % (name, len(a1), len(a0))]
    atol = WEIGHT_ATOL if name.startswith(("hrp", "herc", "nco")) else MATRIX_ATOL
    for k, (x, y) in enumerate(zip(a1, a0)):
        if x.shape != y.shape:
            fails.append("%s[%d]: shape %s, seed %s" % (name, k, x.shape, y.shape))
        elif not np.array_equal(np.isnan(x), np.isnan(y)):
            fails.append("%s[%d]: NaNs moved" % (name, k))
        else:
            d = np.nanmax(np.abs(x - y)) if x.size else 0.0
            if d > atol:
                fails.append("%s[%d]: max abs diff %.3g > %.1g" % (name, k, d, atol))
    return fails
