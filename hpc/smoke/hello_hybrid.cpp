// hello_hybrid.cpp: smoke test for MPI + OpenMP.
// Prints the node and CPU core of every rank and thread, the MPI library
// version and thread-support level, and checks that an Allreduce across all
// ranks (and nodes) gives the right answer.
//   OMP_NUM_THREADS=2 mpirun -np 4 ./hello_hybrid
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#include <sched.h>
#endif

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  char host[MPI_MAX_PROCESSOR_NAME];
  int len = 0;
  MPI_Get_processor_name(host, &len);

  int nthreads = 1;
  std::string where;  // "thread:cpu" pairs; cpu = -1 where sched_getcpu() is unavailable (macOS)
#pragma omp parallel
  {
    int tid = 0, cpu = -1;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#pragma omp single
    nthreads = omp_get_num_threads();
#endif
#ifdef __linux__
    cpu = sched_getcpu();
#endif
#pragma omp critical
    where += (where.empty() ? "" : " ") + std::to_string(tid) + ":" + std::to_string(cpu);
  }

  const int kLine = 1024;
  std::vector<char> line(kLine, '\0');
  std::snprintf(line.data(), kLine, "rank %3d of %d on %s, %d thread(s), thread:cpu = %s", rank, size,
                host, nthreads, where.c_str());
  std::vector<char> all(rank == 0 ? static_cast<size_t>(kLine) * size : 1);
  MPI_Gather(line.data(), kLine, MPI_CHAR, all.data(), kLine, MPI_CHAR, 0, MPI_COMM_WORLD);

  double mine = rank + 1.0, total = 0.0;
  MPI_Allreduce(&mine, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double expect = size * (size + 1) / 2.0;
  // Every rank holds the sum, so every rank returns the verdict: mpirun then exits non-zero on
  // a wrong sum, and a PBS job that runs this test can gate others (qsub -W depend=afterok).
  const bool pass = total == expect;

  if (rank == 0) {
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    int vlen = 0;
    MPI_Get_library_version(version, &vlen);
    const std::string v(version, vlen);
    std::printf("MPI library: %s\n", v.substr(0, v.find('\n')).c_str());
    std::printf("thread support: provided=%d (need >= %d, MPI_THREAD_FUNNELED)\n", provided,
                MPI_THREAD_FUNNELED);
    for (int r = 0; r < size; ++r) std::printf("%s\n", &all[static_cast<size_t>(r) * kLine]);
    std::printf("Allreduce across %d ranks: %s (got %g, expected %g)\n", size,
                pass ? "PASS" : "FAIL", total, expect);
  }
  MPI_Finalize();
  return pass ? 0 : 1;
}
