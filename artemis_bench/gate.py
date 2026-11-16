"""Correctness gate for the mutual-information codependence (Artemis test command).

    python artemis_bench/gate.py            # check this checkout against golden
    python artemis_bench/gate.py --record   # write golden (seed files only)

This process never imports riskfolio. worker.py computes every case in a child
process; this parent compares the outputs with golden data recorded from the
unmodified Riskfolio-Lib 344b899 (7.4.1): mutual-information and
variation-of-information matrices for every binning rule, both normalisations,
DataFrame and ndarray input, tied values and two assets, plus HRP, HERC and NCO
weights that use them.
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

from compare import compare, harness_digest, limit_worker_memory, load_golden, venv_state  # noqa: E402

GOLDEN = os.path.join(HERE, "golden", "gate")
SEED_SHA256 = {
    "riskfolio/src/AuxFunctions.py": "6be25897d065f81c0532167d25fa66f97642266c7da519aae7a1184f6f9cb71f",
    "riskfolio/src/HCPortfolio.py": "5b6a1ac4cbef5e3ef6d1aa7c13353e5d9121b5cbc863f91a4f835f3dafe3791f",
}


def is_seed():
    for rel, want in SEED_SHA256.items():
        with open(os.path.join(ROOT, rel), "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != want:
                return False
    return True


def run_worker(outdir):
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "worker.py"), "gate", outdir],
                       cwd=outdir, capture_output=True, text=True, timeout=1200,
                       preexec_fn=limit_worker_memory)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        sys.exit("gate: worker exited %d" % r.returncode)


def main():
    record = "--record" in sys.argv
    digest, venv = harness_digest(), venv_state()
    golden = None if record else load_golden(GOLDEN)
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
        names = sorted(golden)
        fails = []
        for name in names:
            fails += compare(name, golden, out)
        if harness_digest() != digest:
            fails.append("the harness or its golden data changed during the run")
        if venv_state() != venv:
            fails.append("the benchmark venv changed during the run")
        for f in fails:
            print("FAIL", f, file=sys.stderr)
        print("gate: %d cases, %d failures" % (len(names), len(fails)))
        if fails:
            return 1
    finally:
        shutil.rmtree(out, ignore_errors=True)
    # The repo's own hc_ tests are not run: on the seed with these pins they fail before reaching
    # this code (test_hc_hrp_optimization loops over every risk measure, and EVaR raises
    # UnboundLocalError in RiskFunctions.py without an exponential-cone solver). The cases above
    # cover HRP, HERC and NCO with mutual information directly.
    return 0


if __name__ == "__main__":
    sys.exit(main())
