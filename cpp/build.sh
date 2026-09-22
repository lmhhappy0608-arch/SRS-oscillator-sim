#!/usr/bin/env bash
# Fast build without CMake (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

CXX="${CXX:-clang++}"
FLAGS=(-std=c++17 -O3 -DNDEBUG -I third_party/cnpy)
LDFLAGS=()
OPENMP_ENABLED=0

detect_libomp_prefix() {
  [[ "$(uname)" == "Darwin" ]] || return 1

  local candidates=()
  if command -v brew &>/dev/null; then
    local brew_prefix
    brew_prefix="$(brew --prefix libomp 2>/dev/null || true)"
    if [[ -n "$brew_prefix" && -d "$brew_prefix/lib" ]]; then
      candidates+=("$brew_prefix")
    fi
  fi
  candidates+=("/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")

  local seen="" dir
  for dir in "${candidates[@]}"; do
    [[ " $seen " == *" $dir "* ]] && continue
    seen+=" $dir"
    if [[ -f "$dir/lib/libomp.dylib" || -f "$dir/lib/libomp.a" ]]; then
      echo "$dir"
      return 0
    fi
  done
  return 1
}

if [[ "$(uname)" == "Darwin" ]]; then
  if libomp_prefix="$(detect_libomp_prefix)"; then
    FLAGS+=(-Xpreprocessor -fopenmp -DUSE_OPENMP=1 "-I${libomp_prefix}/include")
    LDFLAGS+=("-L${libomp_prefix}/lib" -Wl,-rpath,"${libomp_prefix}/lib" -lomp)
    OPENMP_ENABLED=1
    echo "OpenMP: enabled (libomp at ${libomp_prefix})"
  else
    echo "warning: libomp not found; building serial. install: brew install libomp" >&2
  fi
elif command -v g++ &>/dev/null && g++ -fopenmp -x c++ - -o /dev/null 2>/dev/null <<< ""; then
  FLAGS+=(-fopenmp -DUSE_OPENMP=1)
  OPENMP_ENABLED=1
  echo "OpenMP: enabled (g++ -fopenmp)"
else
  echo "warning: no OpenMP support detected; building serial." >&2
fi

echo "Building with $CXX ..."
LINK_ARGS=(-lz)
if ((${#LDFLAGS[@]} > 0)); then
  LINK_ARGS+=("${LDFLAGS[@]}")
fi
"$CXX" "${FLAGS[@]}" \
  main.cpp third_party/cnpy/cnpy.cpp \
  "${LINK_ARGS[@]}" \
  -o build/generate_raw_grid

echo "OK: build/generate_raw_grid (OpenMP: $([[ "$OPENMP_ENABLED" -eq 1 ]] && echo enabled || echo disabled))"
