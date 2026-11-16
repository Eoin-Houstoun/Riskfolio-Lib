"""Correctness gate for the mutual-information codependence (Artemis test command).

    python artemis_bench/gate.py            # check this checkout against golden
    python artemis_bench/gate.py --record   # write golden (seed files only)

This process never imports riskfolio. worker.py computes every case in a child
process; this parent compares the outputs with golden data recorded from the
unmodified Riskfolio-Lib 344b899 (7.4.1): mutual-information and
variation-of-information matrices for every binning rule, both normalisations,
DataFrame and ndarray input, tied values and two assets, plus HRP, HERC and NCO
weights that use them. Then it runs the repository's own hierarchical tests.
A missing output or a crash fails.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

from compare import compare, drop_private, harness_digest, private_golden, venv_unchanged  # noqa: E402

GOLDEN = os.path.join(HERE, "golden", "gate")
SEED_SHA256 = {
    "riskfolio/src/AuxFunctions.py": "6be25897d065f81c0532167d25fa66f97642266c7da519aae7a1184f6f9cb71f",
    "riskfolio/src/HCPortfolio.py": "5b6a1ac4cbef5e3ef6d1aa7c13353e5d9121b5cbc863f91a4f835f3dafe3791f",
}
REPO_TESTS = ["tests/test_portfolio.py", "-k", "hc_", "-q", "-p", "no:cacheprovider"]


def is_seed():
    for rel, want in SEED_SHA256.items():
        with open(os.path.join(ROOT, rel), "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != want:
                return False
    return True


def run_worker(outdir):
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "worker.py"), "gate", outdir],
                       cwd=outdir, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        sys.exit("gate: worker exited %d" % r.returncode)


def main():
    record = "--record" in sys.argv
    digest = harness_digest()
    golden = None if record else private_golden(GOLDEN)
    out = tempfile.mkdtemp(prefix="rfgate-")
    try:
        run_worker(out)
        if record:
            if not is_seed():
                sys.exit("gate --record: riskfolio/src is not the seed; refusing")
            errs = [f for f in os.listdir(out) if f.endswith(".err")]
            if errs:
                sys.exit("gate --record: seed raised in %s" % errs)
            shutil.rmtree(GOLDEN, ignore_errors=True)
            shutil.copytree(out, GOLDEN)
            print("recorded %d files into %s" % (len(os.listdir(GOLDEN)), GOLDEN))
            return 0
        names = sorted(f[:-len(".meta.json")] for f in os.listdir(golden) if f.endswith(".meta.json"))
        fails = []
        for name in names:
            fails += compare(name, golden, out)
        if harness_digest() != digest:
            fails.append("the harness or its golden data changed during the run")
        ok, why = venv_unchanged()
        if not ok:
            fails.append("the benchmark venv changed during the run: %s" % why)
        for f in fails:
            print("FAIL", f, file=sys.stderr)
        print("gate: %d cases, %d failures" % (len(names), len(fails)))
        if fails:
            return 1
    finally:
        shutil.rmtree(out, ignore_errors=True)
        if golden:
            drop_private(golden)
    t = subprocess.run([sys.executable, "-m", "pytest", *REPO_TESTS], cwd=ROOT,
                       capture_output=True, text=True, timeout=1800,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg"))
    print("repo tests:", (t.stdout.strip().splitlines() or ["no output"])[-1])
    return 0 if t.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
