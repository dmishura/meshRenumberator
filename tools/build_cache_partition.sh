#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
output=${1:-$root/../experiments/build/cache_partition_backend}
mkdir -p -- "$(dirname -- "$output")"
"${CXX:-c++}" -O3 -std=c++17 -Wall -Wextra -Wpedantic "$root/cache_partition_backend.C" -o "$output"
bash "$root/build_working_set.sh"
