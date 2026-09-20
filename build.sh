#!/usr/bin/env bash
set -euo pipefail
pin=dd6aa230c49082bf4aad4a5a56f513d87a2084ef
test "$(git -C vendor/RaBitQ-Library rev-parse HEAD)" = "$pin"
test -z "$(git -C vendor/RaBitQ-Library status --porcelain)"
cmake -S vendor/RaBitQ-Library -B build-library -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_BUILD_SAMPLES=OFF -DRABITQ_BUILD_TESTS=OFF -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build build-library -j4
g++ -O3 -std=c++17 -fopenmp -Wall -Wextra -Ivendor/RaBitQ-Library/include \
  front_score.cpp build-library/librabitq_core.a -o build-library/front_score
