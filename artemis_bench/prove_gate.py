"""Prove the harness both ways: the seed passes, each injected fault fails.

    python artemis_bench/prove_gate.py          # gate faults
    python artemis_bench/prove_gate.py --bench  # also bench attacks

Patches riskfolio/src/AuxFunctions.py in place one fault at a time and restores
the original bytes afterwards (also on error).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AUX = os.path.join(ROOT, "riskfolio", "src", "AuxFunctions.py")


def first(old, new):
    def f(src):
        assert old in src, "pattern missing: %r" % old
        return src.replace(old, new, 1)
    return f


def both(*patches):
    def f(src):
        for p in patches:
            src = p(src)
        return src
    return f


# Plausible wrong rewrites of the hot loop, each in mutual_info_matrix (the first match).
WRONG_CACHE_KEY = first("/ knuth_bin_width(X1[:, j])", "/ knuth_bin_width(X1[:, i])")
FAULTS = {
    "bin width cached under the wrong column": WRONG_CACHE_KEY,
    "normalised by max entropy, not min": first(
        "iXY = iXY / np.min([hX, hY])", "iXY = iXY / np.max([hX, hY])"),
    "bin count truncated, not rounded": first(
        "bins = np.int32(np.round(np.maximum(k1, k2)))", "bins = np.int32(np.maximum(k1, k2))"),
    "lower triangle not filled": first("        mat[j, i] = mat[i, j]\n", ""),
    "child patches numpy + wrong cache key": both(
        WRONG_CACHE_KEY, lambda s: s + "\nnp.nanmax = lambda *a, **k: 0.0\nnp.abs = lambda x: x * 0\n"),
    "child edits golden + wrong cache key": both(
        WRONG_CACHE_KEY, lambda s: s + (
            "\nimport glob as _g, os as _o\n"
            "for _f in _g.glob(_o.path.join(_o.path.dirname(__file__), '..', '..', 'artemis_bench',"
            " 'golden', 'gate', '*.npz')):\n"
            "    _o.chmod(_f, 0o644); open(_f, 'ab').write(b'x')\n")),
}
REPO_CACHE = first(
    "def mutual_info_matrix(X, bins_info=\"KN\", normalize=True):\n",
    "def mutual_info_matrix(X, bins_info=\"KN\", normalize=True):\n"
    "    import pathlib as _pl\n"
    "    _pl.Path(__file__).with_name('_mi_cache.txt').write_text('x')\n")


def run(script, extra_env=None):
    env = dict(os.environ, **(extra_env or {}))
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, script)],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    lines = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode, lines


def first_fail(lines):
    return next((l for l in lines if l.startswith(("FAIL", "bench:"))), lines[-1] if lines else "")


def main():
    with open(AUX) as fh:
        seed = fh.read()
    golden_dir = os.path.join(HERE, "golden")
    golden_backup = os.path.join(tempfile.mkdtemp(prefix="rfprove-"), "golden")
    shutil.copytree(golden_dir, golden_backup)
    ok = True
    try:
        rc, out = run("gate.py")
        print("gate  seed                                   %s  %s" % ("PASS" if rc == 0 else "FAIL", out[-1]))
        ok &= rc == 0
        for name, patch in FAULTS.items():
            with open(AUX, "w") as fh:
                fh.write(patch(seed))
            rc, out = run("gate.py")
            ok &= rc != 0
            print("gate  fault %-34s %s  %s" % (name, "CAUGHT" if rc else "MISSED", first_fail(out)[:80]))
            with open(AUX, "w") as fh:
                fh.write(seed)
            shutil.rmtree(golden_dir)
            shutil.copytree(golden_backup, golden_dir)

        if "--bench" in sys.argv:
            one = {"BENCH_REPEATS": "1"}
            rc, out = run("bench.py", one)
            ok &= rc == 0
            base = None
            if rc == 0:
                with open(os.path.join(ROOT, "artemis_results.json")) as fh:
                    base = json.load(fh)["hrp_mutual_info_50_assets_s"]
            print("bench seed                                   %s  hrp %.2fs" % ("PASS" if rc == 0 else "FAIL", base or -1))
            for name, patch in (("wrong weights (bin width cached wrong)", WRONG_CACHE_KEY),
                                ("writes a cache file into the repo", REPO_CACHE)):
                with open(AUX, "w") as fh:
                    fh.write(patch(seed))
                rc, out = run("bench.py", one)
                ok &= rc != 0
                print("bench attack %-33s %s  %s" % (name, "CAUGHT" if rc else "MISSED", first_fail(out)[:80]))
                with open(AUX, "w") as fh:
                    fh.write(seed)
                cache = os.path.join(ROOT, "riskfolio", "src", "_mi_cache.txt")
                if os.path.exists(cache):
                    os.remove(cache)
    finally:
        with open(AUX, "w") as fh:
            fh.write(seed)
        shutil.rmtree(golden_dir, ignore_errors=True)
        shutil.copytree(golden_backup, golden_dir)
        shutil.rmtree(os.path.dirname(golden_backup), ignore_errors=True)
    print("ALL AS EXPECTED" if ok else "SOMETHING UNEXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
