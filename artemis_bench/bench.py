"""Benchmark: HRP with mutual-information codependence (Artemis benchmark command).

    python artemis_bench/bench.py            # writes ./artemis_results.json
    python artemis_bench/bench.py --record   # seed files only: golden weights

Headline: HCPortfolio.optimization(model="HRP", codependence="mutual_info",
rm="MV") on 50 assets x 2,520 daily returns (10 years). On the seed about 88%
of it is spent recomputing knuth_bin_width for every asset pair.

How it stays honest:
- This process never imports riskfolio, so code under test cannot patch its
  clock or its checks. Each repeat runs worker.py in a fresh process (no
  caching across repeats); the clock here runs from sending a nonce to reading
  "ARTEMIS_DONE <nonce>" back. Loading inputs and saving outputs are outside.
- One untimed warm-up repeat first, then the median of BENCH_REPEATS (5).
- Every repeat's weights are compared with golden weights recorded from the seed.
- Single-threaded: each child's CPU time over its wall time must stay under 1.3.
- No new files may appear in the repository while it runs.
- A canary (an eigh on a fixed matrix, in this process) shows machine noise.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import secrets  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

from compare import compare  # noqa: E402
from gate import is_seed  # noqa: E402

GOLDEN = os.path.join(HERE, "golden", "bench")
REPEATS = int(os.environ.get("BENCH_REPEATS", "5"))
MAX_CPU_WALL = 1.3


def repo_files():
    seen = set()
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x != ".git"]
        seen.update(os.path.join(d, f) for f in files)
    return seen


def run_once(outdir):
    """One fresh child: returns (timed seconds, child cpu/wall)."""
    # HOME and caches point into this repeat's own folder, deleted after it, so nothing
    # written by one repeat (a results cache under ~/.cache, say) can speed up the next.
    env = dict(os.environ, TMPDIR=outdir, HOME=outdir, XDG_CACHE_HOME=outdir,
               MPLCONFIGDIR=outdir, JOBLIB_TEMP_FOLDER=outdir)
    spawn = time.perf_counter()
    p = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "worker.py"), "hrp", outdir],
                         cwd=outdir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1)
    for line in p.stdout:
        if line.strip() == "ARTEMIS_READY":
            break
    else:
        sys.exit("bench: worker never became ready")
    nonce = secrets.token_hex(16)
    t0 = time.perf_counter()
    p.stdin.write(nonce + "\n")
    p.stdin.flush()
    t1 = None
    for line in p.stdout:
        if line.strip() == "ARTEMIS_DONE " + nonce:
            t1 = time.perf_counter()
            break
    if t1 is None:
        sys.exit("bench: worker ended without finishing the timed call")
    p.stdout.read()
    _, status, ru = os.wait4(p.pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        sys.exit("bench: worker exited %d" % os.waitstatus_to_exitcode(status))
    return t1 - t0, (ru.ru_utime + ru.ru_stime) / (time.perf_counter() - spawn)


def main():
    record = "--record" in sys.argv
    if record and not is_seed():
        sys.exit("bench --record: riskfolio/src is not the seed; refusing")

    before = repo_files()
    rng = np.random.default_rng(20261116)
    canary_m = rng.standard_normal((700, 700))
    canary_m = canary_m @ canary_m.T
    scipy.linalg.eigh(canary_m, eigvals_only=True)  # warm-up

    times, canary, worst_ratio = [], [], 0.0
    for rep in range(1 if record else REPEATS + 1):
        out = tempfile.mkdtemp(prefix="rfbench-")
        try:
            t, ratio = run_once(out)
            if record:
                shutil.rmtree(GOLDEN, ignore_errors=True)
                shutil.copytree(out, GOLDEN)
                continue
            fails = compare("hrp", GOLDEN, out)
            if fails:
                for f in fails:
                    print("FAIL", f, file=sys.stderr)
                sys.exit("bench: result is wrong; not timing a wrong answer")
        finally:
            shutil.rmtree(out, ignore_errors=True)
        if ratio > MAX_CPU_WALL:
            sys.exit("bench: child cpu/wall %.2f > %.2f; the benchmark is single-threaded"
                     % (ratio, MAX_CPU_WALL))
        if rep == 0:
            continue  # warm-up: checked, not timed
        worst_ratio = max(worst_ratio, ratio)
        times.append(t)
        c0 = time.perf_counter()
        scipy.linalg.eigh(canary_m, eigvals_only=True)
        canary.append(time.perf_counter() - c0)

    if record:
        print("recorded", sorted(os.listdir(GOLDEN)))
        return 0
    new = repo_files() - before - {os.path.join(ROOT, "artemis_results.json")}
    if new:
        sys.exit("bench: new files appeared in the repository: %s" % sorted(new)[:5])

    def cv(xs):
        return statistics.stdev(xs) / statistics.mean(xs) if len(xs) > 1 else 0.0

    print("per-repeat hrp    %s" % " ".join("%.3f" % x for x in times), file=sys.stderr)
    print("per-repeat canary %s" % " ".join("%.4f" % x for x in canary), file=sys.stderr)
    res = {
        "hrp_mutual_info_50_assets_s": statistics.median(times),
        "canary_eigh_s": statistics.median(canary),
        "hrp_mutual_info_50_assets_cv": cv(times),
        "canary_eigh_cv": cv(canary),
        "max_child_cpu_over_wall": worst_ratio,
    }
    with open(os.path.join(os.getcwd(), "artemis_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
