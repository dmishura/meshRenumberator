#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
wfgs=$(realpath "${WFGS_ROOT:-$root/../../WavefrontGaussSeidel}")
build=$(realpath "${WFGS_BUILD:-$wfgs/build-cmake}")
output=${1:-$root/../experiments/build/gs_probe}
source=${2:-$root/gs_probe.C}
mkdir -p -- "$(dirname -- "$output")"
compiler=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "$build/CMakeCache.txt")
itt=$(sed -n 's/^ITTNOTIFY_LIBRARY:FILEPATH=//p' "$build/CMakeCache.txt")
extra=()
if [[ -f $itt ]]; then
    extra+=("$itt" -ldl)
fi
"${CXX:-${compiler:-c++}}" -O3 -std=c++17 -fopenmp \
    -I"$wfgs/compat" -I"$wfgs/src" -I"$wfgs/src/GAMG" -I"$wfgs/src/fwGAMG" \
    -I"$wfgs/tests/harness" "$source" \
    "$build/libsmoother_harness.a" "$build/libfw_gamg.a" \
    "$build/libgamg_standalone.a" "$build/libsmoother_kernels.a" \
    "${extra[@]}" -o "$output"
