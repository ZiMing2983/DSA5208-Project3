# Smoke tests for the NUS HPC setup. Also builds on the Mac after `source hpc/env_local.sh`.
#   make cpu   -> hello_hybrid      (MPI + OpenMP)
#   make gpu   -> cuda_aware_check  (MPI + CUDA; on a Vanda GPU node after
#                                    `module load nvhpc-hpcx-cuda12/26.1`)
MPICXX    ?= mpicxx
NVCC      ?= nvcc
# Under the NVIDIA HPC SDK use its CUDA 12 toolkit (HPC-X is built against CUDA 12.9; the
# default nvcc there is CUDA 13.1). See the main Makefile.
NVHPC_CUDA12 := $(if $(NVHPC_ROOT),$(lastword $(sort $(wildcard $(NVHPC_ROOT)/cuda/12.*))))
ifneq (,$(NVHPC_CUDA12))
  NVCC := $(NVHPC_CUDA12)/bin/nvcc
endif
CUDA_ARCH ?= native

# The OpenMP flag depends on the compiler behind mpicxx (override with OMPFLAG=...).
CXX_ID := $(shell $(MPICXX) --version 2>/dev/null | head -1)
ifneq (,$(findstring nvc++,$(CXX_ID)))
  OMPFLAG ?= -mp
else ifneq (,$(findstring Apple,$(CXX_ID)))
  OMPFLAG ?= -Xpreprocessor -fopenmp -lomp
else
  OMPFLAG ?= -fopenmp
endif

# nvcc needs MPI's include and library paths (Open MPI / HPC-X wrapper syntax).
MPI_INC  = $(addprefix -I,$(shell $(MPICXX) --showme:incdirs 2>/dev/null))
MPI_LIBS = $(addprefix -L,$(shell $(MPICXX) --showme:libdirs 2>/dev/null)) \
           $(addprefix -l,$(shell $(MPICXX) --showme:libs 2>/dev/null))

.PHONY: cpu gpu clean
cpu: hello_hybrid
gpu: cuda_aware_check

hello_hybrid: hello_hybrid.cpp
	$(MPICXX) -O2 -std=c++17 $(OMPFLAG) -o $@ $<

cuda_aware_check: cuda_aware_check.cu
	$(NVCC) -O2 -std=c++17 -arch=$(CUDA_ARCH) $(MPI_INC) -o $@ $< $(MPI_LIBS)

clean:
	rm -f hello_hybrid cuda_aware_check
