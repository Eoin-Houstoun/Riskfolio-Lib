#!/bin/sh
# One-time environment for the Riskfolio-Lib HRP mutual-information benchmark.
# Run on the runner host, as the user the runner runs as. Idempotent; pass
# --force to rebuild.
#
# Pinned on purpose (resolved with uv --exclude-newer 2026-10-31): Discovery
# versions write against these APIs, and a system Python silently breaks them.
# vectorbt is left out: riskfolio never imports it. After building, the venv is
# made read-only and a manifest of its files is kept, so code under test cannot
# change the environment later versions use.
set -eu

VENV="${RF_BENCH_VENV:-$HOME/.venvs/riskfolio-hrp-bench}"
PY_VERSION=3.12
PINS="numpy==2.2.6 scipy==1.15.3 pandas==2.2.3 cvxpy==1.9.3 clarabel==0.11.1 scs==3.3.1 osqp==1.1.3
highspy==1.15.1 scikit-learn==1.9.1 statsmodels==0.15.0 arch==8.0.0 xlsxwriter==3.3.1 networkx==3.7
astropy==8.0.2 astropy-iers-data==0.2026.10.19.0.52.1 matplotlib==3.11.2 pybind11==3.1.0
setuptools==84.2.1 pytest==9.2.0"

manifest() {
    # Hash of every entry's type, mode and path, every file's content, and every symlink's
    # target under the venv (minus the manifest itself). POSIX tools only, never the venv's
    # own interpreter. Each step's exit status is checked (sh has no pipefail), so an
    # unreadable file fails instead of hashing less.
    tmp=$(mktemp -d) || return 1
    if (cd "$VENV" \
        && find . -mindepth 1 -path ./.manifest -prune -o -printf '%y %m %p\n' > "$tmp/ls" \
        && find . -path ./.manifest -prune -o -type f -exec sha256sum {} + > "$tmp/files" \
        && find . -type l -exec sh -c 'for l; do printf "L %s -> %s\n" "$l" "$(readlink "$l")"; done' _ {} + > "$tmp/links"); then
        LC_ALL=C sort "$tmp/ls" "$tmp/files" "$tmp/links" | sha256sum | cut -d' ' -f1
        rm -rf "$tmp"
    else
        rm -rf "$tmp"
        echo "setup.sh: could not hash every file in $VENV" >&2
        return 1
    fi
}

check() {
    [ -f "$VENV/.manifest" ] || { echo "no manifest"; return 1; }
    now=$(manifest) || { echo "venv could not be hashed"; return 1; }
    [ "$now" = "$(cat "$VENV/.manifest")" ] || { echo "venv changed since setup (manifest mismatch)"; return 1; }
    "$VENV/bin/python" -I - <<'EOF'
import sys, numpy, scipy, pandas, astropy, sklearn
want = {"python": "3.12", "numpy": "2.2.6", "scipy": "1.15.3", "pandas": "2.2.3",
        "astropy": "8.0.2", "sklearn": "1.9.1"}
have = {"python": "%d.%d" % sys.version_info[:2], "numpy": numpy.__version__,
        "scipy": scipy.__version__, "pandas": pandas.__version__,
        "astropy": astropy.__version__, "sklearn": sklearn.__version__}
bad = {k: (have[k], v) for k, v in want.items() if have[k] != v}
if bad:
    sys.exit("version mismatch (have, want): %r" % bad)
print("ok " + " ".join("%s %s" % kv for kv in have.items()))
EOF
}

if [ "${1:-}" = "--manifest" ]; then
    # gate.py and bench.py record this before a run and compare it after
    manifest
    exit $?
fi

if [ "${1:-}" = "--check" ]; then
    # the venv matches its manifest and its pinned versions
    check
    exit $?
fi

if [ "${1:-}" != "--force" ] && [ -x "$VENV/bin/python" ] && check >/dev/null 2>&1; then
    check
    exit 0
fi

command -v uv >/dev/null 2>&1 || {
    echo "setup.sh: uv not found. Install it (https://docs.astral.sh/uv/) as the runner user." >&2
    echo "setup.sh: refusing to fall back to the system Python." >&2
    exit 1
}
command -v c++ >/dev/null 2>&1 || {
    echo "setup.sh: no C++ compiler (riskfolio builds a pybind11 extension). apt install build-essential" >&2
    exit 1
}

mkdir -p "$(dirname "$VENV")"
exec 9>"$VENV.lock"
flock 9
if [ "${1:-}" != "--force" ] && [ -x "$VENV/bin/python" ] && check >/dev/null 2>&1; then
    check
    exit 0
fi
[ -d "$VENV" ] && chmod -R u+w "$VENV"
rm -rf "$VENV"
uv venv --quiet --python "$PY_VERSION" "$VENV"
# shellcheck disable=SC2086
uv pip install --quiet --python "$VENV/bin/python" $PINS
"$VENV/bin/python" -c "import numpy, scipy, pandas, cvxpy, sklearn, statsmodels, arch, astropy.stats, networkx, matplotlib, xlsxwriter, pybind11, pytest" >/dev/null
# Contents read-only first, then hash them, then lock the top folder and the manifest.
find "$VENV" -mindepth 1 ! -type l -exec chmod a-w {} +  # never follow links out of the venv
manifest > "$VENV/.manifest"
chmod a-w "$VENV/.manifest" "$VENV"
check
