#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
output=${1:-$root/../experiments/build}
mkdir -p -- "$output"
"${CXX:-c++}" -O3 -std=c++17 -Wall -Wextra -Wpedantic \
    "$root/cache_traversal_backend.C" -o "$output/cache_traversal_backend"
bash "$root/build_working_set.sh" "$output/working_set_backend"
bash "$root/build_probe.sh" "$output/gs_layout" "$root/gs_layout.C"
