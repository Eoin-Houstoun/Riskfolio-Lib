"""Child process for gate.py and bench.py. Only this process imports riskfolio.

The parents never import riskfolio, so nothing in the code under test can patch
the parents' clocks or comparison functions. The parents check every output
this process writes against golden data recorded from the seed.

    worker.py gate <outdir>    compute every gate case, save outputs
    worker.py hrp <outdir>     one benchmark call:
        prints ARTEMIS_READY, reads a nonce line from stdin, runs the timed
        call, prints "ARTEMIS_DONE <nonce>", then saves the outputs.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import riskfolio as rp  # noqa: E402
import riskfolio.src.AuxFunctions as af  # noqa: E402

if not os.path.abspath(rp.__file__).startswith(ROOT + os.sep):
    sys.exit("worker: imported riskfolio from %s, not this checkout" % rp.__file__)
warnings.filterwarnings("ignore")


def synthetic(n=None, rows=None):
    df = pd.read_csv(os.path.join(HERE, "data", "returns50.csv.gz"), index_col=0, parse_dates=True)
    if rows:
        df = df.iloc[-rows:]
    return df.iloc[:, :n] if n else df


def repo_returns():
    # The repository's own test prices, as its HRP test reads them.
    px = pd.read_csv(os.path.join(ROOT, "tests", "stock_prices.csv"), index_col=0, parse_dates=True)
    return px.pct_change().dropna()


def hc_weights(Y, model, rm, with_codep=False, **kw):
    port = rp.HCPortfolio(returns=Y)
    w = port.optimization(model=model, codependence="mutual_info", rm=rm, rf=0,
                          linkage="single", max_k=10, leaf_order=True, **kw)
    # The call also computes and keeps the mutual-information matrix (port.codep), which the
    # weights alone don't reveal; the benchmark checks both.
    return (w, port.codep) if with_codep else w


def gate_cases():
    syn20 = synthetic(20, 1000)
    rep = repo_returns()
    # Matrices: every binning rule, both normalisations, DataFrame and ndarray input.
    # Not HGR: on the diagonal the seed tests corr == 1, np.corrcoef of a column with itself
    # gives 0.9999999999999999 for two of these assets, and the seed then asks histogram2d
    # for 72,200 x 72,200 bins (about 42 GB). The seed itself cannot run it on this data.
    for b in ("KN", "FD", "SC", 10, np.int32(7)):
        tag = b if isinstance(b, str) else "int%d" % int(b)
        yield "mi_repo_%s" % tag, lambda b=b: af.mutual_info_matrix(rep, bins_info=b)
        yield "vi_repo_%s" % tag, lambda b=b: af.var_info_matrix(rep, bins_info=b)
    yield "mi_syn20_KN", lambda: af.mutual_info_matrix(syn20)
    yield "vi_syn20_KN", lambda: af.var_info_matrix(syn20)
    yield "mi_syn20_raw", lambda: af.mutual_info_matrix(syn20, normalize=False)
    yield "vi_syn20_raw", lambda: af.var_info_matrix(syn20, normalize=False)
    yield "mi_syn20_ndarray", lambda: af.mutual_info_matrix(syn20.to_numpy())
    yield "vi_syn20_ndarray", lambda: af.var_info_matrix(syn20.to_numpy())
    yield "mi_two_assets", lambda: af.mutual_info_matrix(synthetic(2, 300))
    yield "vi_tied_values", lambda: af.var_info_matrix(synthetic(10, 800))  # A07 has many ties
    # The public entry points that use them.
    yield "codep_dist_repo", lambda: af.codep_dist(rep, codependence="mutual_info")
    yield "hrp_repo_MV", lambda: hc_weights(rep, "HRP", "MV")
    yield "hrp_repo_CVaR", lambda: hc_weights(rep, "HRP", "CVaR")
    yield "herc_repo_MV", lambda: hc_weights(rep, "HERC", "MV")
    yield "nco_repo_MV", lambda: hc_weights(rep, "NCO", "MV")
    yield "hrp_syn20_FD", lambda: hc_weights(syn20, "HRP", "MV", bins_info="FD")
    yield "hrp_syn20_KN", lambda: hc_weights(syn20, "HRP", "MV")


def save(outdir, name, value):
    """Save a matrix, a weights frame or a (codep, dist) pair as .npz plus its kind."""
    if isinstance(value, tuple):
        arrays = {"a%d" % i: np.asarray(v, dtype=float) for i, v in enumerate(value)}
        kind = ["%s %s" % (type(v).__name__, np.asarray(v).dtype) for v in value]
        labels = [list(map(str, getattr(v, "index", []))) for v in value]
    else:
        arrays = {"a0": np.asarray(value, dtype=float)}
        kind = ["%s %s" % (type(value).__name__, np.asarray(value).dtype)]
        labels = [list(map(str, getattr(value, "index", [])))]
    np.savez(os.path.join(outdir, name + ".npz"), **arrays)
    with open(os.path.join(outdir, name + ".meta.json"), "w") as fh:
        json.dump({"kind": kind, "labels": labels}, fh)


def main():
    mode, outdir = sys.argv[1], sys.argv[2]
    if mode == "gate":
        for name, fn in gate_cases():
            try:
                save(outdir, name, fn())
            except Exception:
                with open(os.path.join(outdir, name + ".err"), "w") as fh:
                    fh.write(traceback.format_exc())
        return 0
    if mode != "hrp":
        sys.exit("worker: unknown mode %r" % mode)
    Y = synthetic()
    print("ARTEMIS_READY", flush=True)
    nonce = sys.stdin.readline().strip()
    result = hc_weights(Y, "HRP", "MV", with_codep=True)
    print("ARTEMIS_DONE " + nonce, flush=True)
    save(outdir, "hrp", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
