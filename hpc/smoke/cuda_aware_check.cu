// cuda_aware_check.cu: smoke test for MPI + CUDA on Vanda.
//   1. Each rank picks a GPU (one per rank on a node; round-robin if there are fewer GPUs).
//   2. Checks whether the MPI library is CUDA-aware, i.e. accepts GPU pointers.
//   3. Sends a GPU buffer from rank 0 to rank 1 through MPI and verifies it.
//   4. Times a ping-pong two ways: GPU pointers passed straight to MPI
//      (CUDA-aware), and host staging (cudaMemcpy -> MPI -> cudaMemcpy).
// Run with 2 ranks:  mpirun -np 2 ./cuda_aware_check
// Pass --force to run the GPU-pointer tests when support is reported as "unknown".
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>  // Open MPI / HPC-X: MPIX_CUDA_AWARE_SUPPORT, MPIX_Query_cuda_support()
#endif
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                                   \
  do {                                                                                     \
    cudaError_t err_ = (call);                                                             \
    if (err_ != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                   __LINE__);                                                              \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                        \
    }                                                                                      \
  } while (0)

__global__ void fill(double* a, size_t n) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) a[i] = 1.0 + static_cast<double>(i);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const bool force = argc > 1 && std::strcmp(argv[1], "--force") == 0;

  // 1. GPU selection by node-local rank.
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
  int local = 0;
  MPI_Comm_rank(node, &local);
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 1) {
    std::fprintf(stderr, "rank %d: no GPU visible\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const int dev = local % ndev;
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  char host[MPI_MAX_PROCESSOR_NAME];
  int len = 0;
  MPI_Get_processor_name(host, &len);
  std::printf("rank %d on %s: GPU %d of %d = %s, %.1f GB, compute capability %d.%d\n", rank, host,
              dev, ndev, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
  std::fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  // 2. Is MPI CUDA-aware?  1 = yes, 0 = no, -1 = unknown (no Open MPI extension header).
  int built = -1, runtime = -1;
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  built = MPIX_CUDA_AWARE_SUPPORT;
  runtime = MPIX_Query_cuda_support();
#endif
  if (rank == 0)
    std::printf("CUDA-aware MPI: built-in=%d, at run time=%d  (1 yes, 0 no, -1 unknown)\n", built,
                runtime);
  const bool aware = runtime == 1 || (runtime == -1 && force);

  if (size < 2) {
    if (rank == 0) std::printf("need at least 2 ranks for the transfer tests\n");
    MPI_Comm_free(&node);
    MPI_Finalize();
    return 0;
  }

  const size_t max_bytes = static_cast<size_t>(64) << 20;  // 64 MiB
  const size_t max_n = max_bytes / sizeof(double);
  double* d_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, max_bytes));
  std::vector<double> h_buf(max_n);
  const int peer = rank == 0 ? 1 : 0;
  const bool active = rank < 2;  // extra ranks only join the barriers

  // 3. Correctness: GPU buffer rank 0 -> rank 1.
  if (aware) {
    int ok = 1;
    if (rank == 0) {
      fill<<<static_cast<unsigned>((max_n + 255) / 256), 256>>>(d_buf, max_n);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Send(d_buf, static_cast<int>(max_n), MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
      CUDA_CHECK(cudaMemset(d_buf, 0, max_bytes));
      MPI_Recv(d_buf, static_cast<int>(max_n), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      CUDA_CHECK(cudaMemcpy(h_buf.data(), d_buf, max_bytes, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < max_n; ++i) {
        if (h_buf[i] != 1.0 + static_cast<double>(i)) {
          ok = 0;
          break;
        }
      }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 1, MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("GPU buffer sent rank 0 -> 1 through MPI: %s\n", ok ? "PASS" : "FAIL");
  } else if (rank == 0) {
    std::printf("skipping GPU-pointer tests (MPI not CUDA-aware; pass --force to try anyway)\n");
  }

  // 4. Ping-pong: returns the one-way time per message in seconds.
  auto pingpong = [&](bool device_ptrs, size_t bytes, int reps) {
    const int n = static_cast<int>(bytes / sizeof(double));
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int r = 0; r < reps && active; ++r) {
      for (int leg = 0; leg < 2; ++leg) {
        const bool sending = (leg == 0) == (rank == 0);  // leg 0: 0 -> 1, leg 1: 1 -> 0
        if (device_ptrs) {
          if (sending)
            MPI_Send(d_buf, n, MPI_DOUBLE, peer, 1, MPI_COMM_WORLD);
          else
            MPI_Recv(d_buf, n, MPI_DOUBLE, peer, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (sending) {
          CUDA_CHECK(cudaMemcpy(h_buf.data(), d_buf, bytes, cudaMemcpyDeviceToHost));
          MPI_Send(h_buf.data(), n, MPI_DOUBLE, peer, 1, MPI_COMM_WORLD);
        } else {
          MPI_Recv(h_buf.data(), n, MPI_DOUBLE, peer, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          CUDA_CHECK(cudaMemcpy(d_buf, h_buf.data(), bytes, cudaMemcpyHostToDevice));
        }
      }
    }
    const double t = (MPI_Wtime() - t0) / (2.0 * reps);
    MPI_Barrier(MPI_COMM_WORLD);
    return t;
  };

  if (rank == 0)
    std::printf("\n%12s %12s %12s %12s %12s\n", "bytes", "aware_us", "aware_GB/s", "staged_us",
                "staged_GB/s");
  const size_t sizes[] = {8, 64, 512, 4096, 32768, 262144, 2097152, 16777216, 67108864};
  for (size_t bytes : sizes) {
    const int reps = bytes <= 32768 ? 200 : (bytes <= 2097152 ? 50 : 10);
    double ta = -1.0;
    if (aware) {
      pingpong(true, bytes, 2);  // warm-up
      ta = pingpong(true, bytes, reps);
    }
    pingpong(false, bytes, 2);  // warm-up
    const double ts = pingpong(false, bytes, reps);
    if (rank == 0) {
      if (ta > 0)
        std::printf("%12zu %12.2f %12.2f", bytes, ta * 1e6, bytes / ta / 1e9);
      else
        std::printf("%12zu %12s %12s", bytes, "-", "-");
      std::printf(" %12.2f %12.2f\n", ts * 1e6, bytes / ts / 1e9);
    }
  }

  CUDA_CHECK(cudaFree(d_buf));
  MPI_Comm_free(&node);
  MPI_Finalize();
  return 0;
}
