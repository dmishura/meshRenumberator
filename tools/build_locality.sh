#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
output=${1:-$root/../experiments/build/locality_optimizer}
mkdir -p -- "$(dirname -- "$output")"
"${CXX:-c++}" -O3 -std=c++17 -Wall -Wextra -Wpedantic "$root/locality_optimizer.C" -o "$output"
