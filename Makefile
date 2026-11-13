# DSA5208 Project 3: matrix-free CG for the 2D/3D Poisson equation, serial / MPI+OpenMP / MPI+CUDA.
#
#   make              build/poisson_serial and build/poisson_mpi (MPI + OpenMP)
#   make cuda         build/poisson_cuda (MPI + CUDA; needs nvcc, CUDA_ARCH defaults to the A40)
#   make test         local validation V1, V2a, V2b, V3 (tests/validate.sh)
#   make clean
#
# Mac: `source hpc/env_local.sh` first.  Clusters: build on a compute node (-march=native),
# e.g. `make BUILD=build/$PBS_JOBID` inside the job script.

BUILD     ?= build
MPICXX    ?= mpicxx
NVCC      ?= nvcc
CUDA_ARCH ?= sm_86

# The serial build uses the compiler behind mpicxx, so p = 1 compares like with like.
MPI_UNDERLYING := $(firstword $(shell $(MPICXX) --showme:command 2>/dev/null || $(MPICXX) -show 2>/dev/null))
SERIAL_CXX ?= $(if $(MPI_UNDERLYING),$(MPI_UNDERLYING),$(CXX))
# First non-empty line of --version (nvc++ prints an empty line first).
CXX_ID := $(shell $(SERIAL_CXX) --version 2>/dev/null | grep -m1 .)

# -ffp-contract=off: no fused multiply-adds, so the stencil gives bit-identical results in every
# build and decomposition (validation test V2a). The solver is memory-bound; FMAs would not help.
ifneq (,$(findstring nvc++,$(CXX_ID) $(SERIAL_CXX)))
  ARCHFLAGS   ?=
  FPFLAGS     ?= -Mnofma -Kieee
  OMP_CFLAGS  ?= -mp
  OMP_LIBS    ?= -mp
  SIMD_CFLAGS ?=
  DEPFLAGS    ?=
  WARN        ?=
else ifneq (,$(findstring Apple,$(CXX_ID)))
  ARCHFLAGS   ?= $(if $(filter arm64,$(shell uname -m)),-mcpu=native,-march=native)
  FPFLAGS     ?= -ffp-contract=off
  OMP_CFLAGS  ?= -Xpreprocessor -fopenmp
  OMP_LIBS    ?= -lomp
  SIMD_CFLAGS ?= -fopenmp-simd
else
  ARCHFLAGS   ?= -march=native
  FPFLAGS     ?= -ffp-contract=off
  OMP_CFLAGS  ?= -fopenmp
  OMP_LIBS    ?= -fopenmp
  SIMD_CFLAGS ?= -fopenmp-simd
endif

OPT      ?= -O3
WARN     ?= -Wall -Wextra
DEPFLAGS ?= -MMD -MP
CXXFLAGS_ALL := -std=c++17 $(OPT) $(ARCHFLAGS) $(FPFLAGS) -Iinclude $(DEPFLAGS)

COMMON := src/grid.cpp src/kernels.cpp src/cg.cpp src/options.cpp
SERIAL_OBJ := $(patsubst src/%.cpp,$(BUILD)/serial/%.o,$(COMMON) src/serial_main.cpp)
MPI_OBJ    := $(patsubst src/%.cpp,$(BUILD)/mpi/%.o,$(COMMON) src/halo.cpp src/mpi_main.cpp)
CUDA_OBJ   := $(patsubst src/%.cpp,$(BUILD)/cuda/%.o,$(COMMON) src/halo.cpp src/cuda/mpi_cuda_main.cpp) \
              $(BUILD)/cuda/cuda/kernels.o

.PHONY: all serial mpi cuda test clean info
all: serial mpi
serial: $(BUILD)/poisson_serial
mpi: $(BUILD)/poisson_mpi
cuda: $(BUILD)/poisson_cuda

info:
	@echo "serial compiler: $(SERIAL_CXX) ($(CXX_ID))"
	@echo "mpi wrapper:     $(MPICXX)"
	@echo "flags:           $(CXXFLAGS_ALL)"
	@echo "openmp:          $(OMP_CFLAGS) | $(OMP_LIBS)"
	@echo "cuda:            $(NVCC) (CUDA_HOME=$(CUDA_HOME), arch $(CUDA_ARCH))"

# --- serial: no MPI, no OpenMP threads (SIMD pragmas only) --------------------------------
$(BUILD)/poisson_serial: $(SERIAL_OBJ)
	$(SERIAL_CXX) $(CXXFLAGS_ALL) $(SIMD_CFLAGS) -o $@ $^

$(BUILD)/serial/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(SERIAL_CXX) $(CXXFLAGS_ALL) $(WARN) $(SIMD_CFLAGS) -c -o $@ $<

# --- MPI + OpenMP -------------------------------------------------------------------------
$(BUILD)/poisson_mpi: $(MPI_OBJ)
	$(MPICXX) $(CXXFLAGS_ALL) $(OMP_CFLAGS) -o $@ $^ $(OMP_LIBS)

$(BUILD)/mpi/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS_ALL) $(WARN) $(OMP_CFLAGS) -c -o $@ $<

# --- MPI + CUDA: kernels via nvcc, host code via mpicxx, linked by mpicxx -----------------
# CUDA toolkit. Under the NVIDIA HPC SDK (Vanda: nvhpc-hpcx-cuda12/26.1, which sets
# NVHPC_ROOT) use the SDK's bundled CUDA 12 toolkit: its HPC-X MPI was built against CUDA 12.9,
# while the SDK's default nvcc is CUDA 13.1, which also needs a newer GPU driver. Elsewhere
# (e.g. Atlas's CUDA/12.x modules) use the toolkit that contains the nvcc on PATH.
NVHPC_CUDA12 := $(if $(NVHPC_ROOT),$(lastword $(sort $(wildcard $(NVHPC_ROOT)/cuda/12.*))))
ifneq (,$(NVHPC_CUDA12))
  CUDA_HOME ?= $(NVHPC_CUDA12)
  NVCC := $(CUDA_HOME)/bin/nvcc
endif
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell command -v $(NVCC) 2>/dev/null))
CUDA_LIBDIR ?= $(firstword $(wildcard $(CUDA_HOME)/lib64 $(CUDA_HOME)/lib))
# --fmad=false keeps the GPU stencil bit-identical to the CPU one (same reasoning as above).
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) --fmad=false -Iinclude

$(BUILD)/poisson_cuda: $(CUDA_OBJ)
	$(MPICXX) $(CXXFLAGS_ALL) -o $@ $^ -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart

$(BUILD)/cuda/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS_ALL) $(WARN) $(SIMD_CFLAGS) -I$(CUDA_HOME)/include -c -o $@ $<

$(BUILD)/cuda/cuda/kernels.o: src/cuda/kernels.cu include/cuda_kernels.hpp include/grid.hpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

test: all
	bash tests/validate.sh $(BUILD)

clean:
	rm -rf $(BUILD)

-include $(wildcard $(BUILD)/*/*.d $(BUILD)/*/*/*.d)
