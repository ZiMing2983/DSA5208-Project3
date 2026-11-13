# Source this on your Mac before building MPI code locally:
#     source hpc/env_local.sh
#
# Anaconda's mpicc/mpicxx (MPICH 4.3) call a conda compiler,
# arm64-apple-darwin20.0.0-clang++, which is not installed.
# These variables point them at Apple's clang/clang++ instead.
# For OpenMP, Apple clang uses Anaconda's libomp and omp.h, which mpicxx
# already has on its include and library paths.
export MPICH_CC=clang
export MPICH_CXX=clang++
export OMPFLAG="-Xpreprocessor -fopenmp -lomp"
echo "local MPI build: $(mpicxx --version 2>/dev/null | head -1) | OpenMP flag: $OMPFLAG"
