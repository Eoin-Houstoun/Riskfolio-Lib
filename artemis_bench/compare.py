"""Comparison for the parents (gate.py, bench.py). Never imports riskfolio.

The seed rounds both matrices to 8 decimals, so a faithful rewrite that sums in
another order can land one rounding unit away; that is allowed (1.5e-8), and
nothing more. Weights follow from the matrices and are held to 1e-9.
"""
import hashlib
import json
import os
import subprocess

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


WORKER_MEMORY = 4 << 30  # bytes of address space a worker may map


def limit_worker_memory():
    """preexec_fn for every worker: a hard address-space cap, so a runaway case fails
    with MemoryError instead of exhausting the machine."""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY, WORKER_MEMORY))


def venv_state():
    """Fail unless this interpreter is the benchmark venv, it still matches its stored
    manifest and pinned versions (setup.sh --check), and it can be hashed in full.
    Returns the content hash, which callers compare before and after a run."""
    import sys
    want = os.path.realpath(os.environ.get("RF_BENCH_VENV") or
                            os.path.expanduser("~/.venvs/riskfolio-hrp-bench"))
    if os.path.realpath(sys.prefix) != want:
        sys.exit("harness: running under %s, not the benchmark venv %s" % (sys.prefix, want))
    setup = os.path.join(HERE, "setup.sh")
    chk = subprocess.run(["sh", setup, "--check"], capture_output=True, text=True)
    if chk.returncode != 0:
        sys.exit("harness: venv check failed: %s" % (chk.stdout + chk.stderr).strip()[-300:])
    man = subprocess.run(["sh", setup, "--manifest"], capture_output=True, text=True)
    if man.returncode != 0 or len(man.stdout.strip()) != 64:
        sys.exit("harness: venv could not be hashed: %s" % man.stderr.strip()[-300:])
    return man.stdout.strip()


MATRIX_ATOL = 1.5e-8   # one unit in the 8th decimal the seed rounds to
WEIGHT_ATOL = 1e-9


def load(d, name):
    with open(os.path.join(d, name + ".meta.json")) as fh:
        meta = json.load(fh)
    with np.load(os.path.join(d, name + ".npz")) as z:
        arrays = [z["a%d" % i].copy() for i in range(len(z.files))]
    return meta, arrays


def load_golden(d):
    """Every golden case, read into this process's memory before any worker starts."""
    return {f[:-len(".meta.json")]: load(d, f[:-len(".meta.json")])
            for f in os.listdir(d) if f.endswith(".meta.json")}


def compare(name, golden, out_dir):
    """List of failure strings for one case (empty list means pass); golden from load_golden."""
    if os.path.exists(os.path.join(out_dir, name + ".err")):
        with open(os.path.join(out_dir, name + ".err")) as fh:
            return ["%s: raised %s" % (name, fh.read().strip().splitlines()[-1])]
    try:
        m1, a1 = load(out_dir, name)
    except FileNotFoundError:
        return ["%s: no output written" % name]
    m0, a0 = golden[name]
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
