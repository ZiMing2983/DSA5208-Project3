// MPI + CUDA driver: one MPI rank per GPU, every vector resident on the device for the whole
// solve. The halo exchange passes device buffers straight to MPI (CUDA-aware, --gpu-mpi aware)
// or copies them through pinned host buffers (--gpu-mpi staged).
//   mpirun -np 2 poisson_cuda --dim 3 --n 512 --iters 200 --gpu-mpi aware
//   mpirun -np 2 poisson_cuda --dim 3 --nx 37 --ny 29 --nz 23 --test apply
// Host code only (compiled by mpicxx); the kernels live in kernels.cu.
#include <cuda_runtime.h>
#include <mpi.h>
#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>  // Open MPI: MPIX_Query_cuda_support
#endif
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cg.hpp"
#include "cuda_kernels.hpp"
#include "grid.hpp"
#include "halo.hpp"
#include "kernels.hpp"
#include "options.hpp"

#define CUDA_CHECK(call)                                                               \
  do {                                                                                 \
    const cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess) {                                                           \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                   __LINE__);                                                          \
      MPI_Abort(MPI_COMM_WORLD, 3);                                                    \
    }                                                                                  \
  } while (0)

namespace {

// Pick the GPU before MPI_Init when the launcher says which local rank we are (CUDA-aware
// transports prefer the device to be set first); otherwise after MPI_Init.
int local_rank_from_env() {
  for (const char* v : {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID",
                        "PMI_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"})
    if (const char* s = std::getenv(v)) return std::atoi(s);
  return -1;
}

// May run before MPI_Init, so it reports errors with exit() rather than MPI_Abort().
int select_device(int local) {
  int ndev = 0;
  const cudaError_t e = cudaGetDeviceCount(&ndev);
  if (e != cudaSuccess || ndev < 1) {
    std::fprintf(stderr, "no CUDA device visible (%s)\n", cudaGetErrorString(e));
    std::exit(3);
  }
  const int dev = local % ndev;
  if (cudaSetDevice(dev) != cudaSuccess) {
    std::fprintf(stderr, "cudaSetDevice(%d) failed\n", dev);
    std::exit(3);
  }
  return dev;
}

int cuda_aware_at_runtime() {  // 1 yes, 0 no, -1 unknown
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  return MPIX_Query_cuda_support();
#else
  return -1;
#endif
}

struct DeviceVec {
  double* p = nullptr;
  std::size_t bytes = 0;
  explicit DeviceVec(const Block& b) : bytes(b.size * sizeof(double)) {
    CUDA_CHECK(cudaMalloc(&p, bytes));
    CUDA_CHECK(cudaMemset(p, 0, bytes));
  }
  ~DeviceVec() { cudaFree(p); }
  DeviceVec(const DeviceVec&) = delete;
  DeviceVec& operator=(const DeviceVec&) = delete;
};

// Blocking halo exchange of a device vector: pack kernels -> MPI -> unpack kernels.
class GpuHalo {
 public:
  GpuHalo(const Decomp& d, bool staged) : d_(d), b_(dev_block(d.block)), staged_(staged) {
    for (int k = 0; k < d.block.dim; ++k)
      for (int s = 0; s < 2; ++s) {
        if (!has(k, s)) continue;
        const std::size_t bytes = face_size(d.block, k) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&dsend_[k][s], bytes));
        CUDA_CHECK(cudaMalloc(&drecv_[k][s], bytes));
        if (staged_) {
          CUDA_CHECK(cudaMallocHost(&hsend_[k][s], bytes));
          CUDA_CHECK(cudaMallocHost(&hrecv_[k][s], bytes));
        }
      }
  }
  ~GpuHalo() {
    for (int k = 0; k < 3; ++k)
      for (int s = 0; s < 2; ++s) {
        if (dsend_[k][s]) cudaFree(dsend_[k][s]);
        if (drecv_[k][s]) cudaFree(drecv_[k][s]);
        if (hsend_[k][s]) cudaFreeHost(hsend_[k][s]);
        if (hrecv_[k][s]) cudaFreeHost(hrecv_[k][s]);
      }
  }
  GpuHalo(const GpuHalo&) = delete;
  GpuHalo& operator=(const GpuHalo&) = delete;

  void exchange(double* u) {
    const Block& b = d_.block;
    for (int k = 0; k < b.dim; ++k)
      for (int s = 0; s < 2; ++s)
        if (has(k, s)) gpu_pack(b_, k, s == 0 ? 1 : b.n[k], u, dsend_[k][s]);
    gpu_check("pack");
    CUDA_CHECK(cudaDeviceSynchronize());  // MPI may read the buffers only after packing
    if (staged_)
      for (int k = 0; k < b.dim; ++k)
        for (int s = 0; s < 2; ++s)
          if (has(k, s))
            CUDA_CHECK(cudaMemcpy(hsend_[k][s], dsend_[k][s], bytes(k), cudaMemcpyDeviceToHost));
    int nreq = 0;
    for (int k = 0; k < b.dim; ++k)
      for (int s = 0; s < 2; ++s)
        if (has(k, s))
          MPI_Irecv(staged_ ? hrecv_[k][s] : drecv_[k][s], (int)face_size(b, k), MPI_DOUBLE,
                    d_.nbr[k][s], 2 * k + (1 - s), d_.cart, &req_[nreq++]);
    for (int k = 0; k < b.dim; ++k)
      for (int s = 0; s < 2; ++s)
        if (has(k, s))
          MPI_Isend(staged_ ? hsend_[k][s] : dsend_[k][s], (int)face_size(b, k), MPI_DOUBLE,
                    d_.nbr[k][s], 2 * k + s, d_.cart, &req_[nreq++]);
    MPI_Waitall(nreq, req_, MPI_STATUSES_IGNORE);
    for (int k = 0; k < b.dim; ++k)
      for (int s = 0; s < 2; ++s)
        if (has(k, s)) {
          if (staged_)
            CUDA_CHECK(cudaMemcpy(drecv_[k][s], hrecv_[k][s], bytes(k), cudaMemcpyHostToDevice));
          gpu_unpack(b_, k, s == 0 ? 0 : b.n[k] + 1, drecv_[k][s], u);
        }
    gpu_check("unpack");
    CUDA_CHECK(cudaDeviceSynchronize());
  }

 private:
  bool has(int k, int s) const { return d_.nbr[k][s] != MPI_PROC_NULL; }
  std::size_t bytes(int k) const { return face_size(d_.block, k) * sizeof(double); }
  const Decomp& d_;
  DevBlock b_;
  bool staged_;
  double* dsend_[3][2] = {};
  double* drecv_[3][2] = {};
  double* hsend_[3][2] = {};
  double* hrecv_[3][2] = {};
  MPI_Request req_[12];
};

// Standard CG on the device. Scalars come back to the host once per dot product (that copy is
// the synchronisation point anyway) and are summed over ranks with MPI_Allreduce.
CGStats gpu_cg(const Grid& g, const Decomp& d, GpuHalo& halo, const double* rhs, double* x,
               const CGOptions& opt) {
  const Block& blk = d.block;
  const DevBlock b = dev_block(blk);
  const DevCoef c = dev_coef(g);
  const std::size_t bytes = blk.size * sizeof(double);
  DeviceVec r(blk), p(blk), q(blk);
  double* d_red = nullptr;
  CUDA_CHECK(cudaMalloc(&d_red, sizeof(double)));
  auto ddot = [&](const double* u, const double* v) {
    CUDA_CHECK(cudaMemset(d_red, 0, sizeof(double)));
    gpu_dot(b, u, v, d_red);
    double h = 0.0;
    CUDA_CHECK(cudaMemcpy(&h, d_red, sizeof(double), cudaMemcpyDeviceToHost));
    return h;
  };
  auto allreduce = [&](double v) {
    MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, d.cart);
    return v;
  };
  auto restart = [&]() {
    CUDA_CHECK(cudaMemset(x, 0, bytes));
    CUDA_CHECK(cudaMemcpy(r.p, rhs, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(p.p, rhs, bytes, cudaMemcpyDeviceToDevice));
  };
  const bool fixed = !(opt.tol > 0);
  const int warm = fixed ? opt.warmup : 0;
  const int total = warm + opt.iters;
  restart();
  double rho = allreduce(ddot(r.p, r.p));
  const double rho0 = rho, bnorm = std::sqrt(rho);

  CGStats st, discard;
  CGStats* T = (warm > 0) ? &discard : &st;
  if (bnorm == 0.0) {
    cudaFree(d_red);
    return st;
  }
  MPI_Barrier(d.cart);
  double t_start = MPI_Wtime(), rel = 1.0;
  for (int it = 0; it < total; ++it) {
    if (it == warm && warm > 0) {
      T = &st;
      MPI_Barrier(d.cart);
      t_start = MPI_Wtime();
    }
    const double t0 = MPI_Wtime();
    halo.exchange(p.p);
    const double t1 = MPI_Wtime();
    gpu_apply_A(b, c, p.p, q.p);
    gpu_check("apply_A");
    const double pq_local = ddot(p.p, q.p);
    const double t2 = MPI_Wtime();
    const double pq = allreduce(pq_local);
    const double t3 = MPI_Wtime();
    const double alpha = rho / pq;
    CUDA_CHECK(cudaMemset(d_red, 0, sizeof(double)));
    gpu_update_xr(b, alpha, p.p, q.p, x, r.p, d_red);
    gpu_check("update_xr");
    double rr = 0.0;
    CUDA_CHECK(cudaMemcpy(&rr, d_red, sizeof(double), cudaMemcpyDeviceToHost));
    const double t4 = MPI_Wtime();
    rr = allreduce(rr);
    const double t5 = MPI_Wtime();
    T->t_halo += t1 - t0;
    T->t_comp += (t2 - t1) + (t4 - t3);
    T->t_reduce += (t3 - t2) + (t5 - t4);
    if (it >= warm) ++st.iters;
    rel = std::sqrt(rr) / bnorm;
    if (!fixed && rel < opt.tol) break;
    const double t6 = MPI_Wtime();
    if (fixed && rel < 1e-13) {  // same restart rule as the CPU solver
      restart();
      rho = rho0;
    } else {
      gpu_update_p(b, rr / rho, r.p, p.p);
      gpu_check("update_p");
      rho = rr;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    T->t_comp += MPI_Wtime() - t6;
  }
  st.t_total = MPI_Wtime() - t_start;
  st.rel_res = rel;
  cudaFree(d_red);
  return st;
}

struct Problem {
  Grid g;
  Decomp d;
};

Problem setup(const Options& o, int nx, int ny, int nz) {
  int p = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  int dims[3];
  decomp_dims(p, o.dim, o.decomp, dims);
  if (o.nlocal > 0) {
    nx = o.nlocal * dims[0];
    ny = o.nlocal * dims[1];
    nz = o.nlocal * dims[2];
  }
  Problem pr;
  pr.g = make_grid(o.dim, nx, ny, nz);
  pr.d = make_decomp(pr.g, MPI_COMM_WORLD, dims);
  return pr;
}

CGOptions cg_options(const Options& o) {
  CGOptions c;
  c.fused = false;
  c.tol = o.tol;
  c.iters = o.iters;
  c.warmup = o.warmup;
  return c;
}

struct Solve {
  CGStats st;
  double err = 0.0;
  double t_h2d = 0.0, t_d2h = 0.0;  // one-off copies of b to and x from the device (max over ranks)
  std::vector<double> x_host;  // this rank's block (for gathering)
};

// Times a host<->device copy with CUDA events (Lecture 10), in seconds.
double timed_copy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind) {
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0, 0));
  CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
  CUDA_CHECK(cudaEventRecord(e1, 0));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / 1e3;
}

Solve gpu_solve(const Options& o, Problem& pr) {
  const Block& b = pr.d.block;
  std::vector<double> rhs_h(b.size, 0.0);
  fill_rhs(pr.g, b, rhs_h.data());
  DeviceVec rhs(b), x(b);
  Solve s;
  // The right-hand side goes to the device once; every CG vector then stays there.
  s.t_h2d = timed_copy(rhs.p, rhs_h.data(), rhs.bytes, cudaMemcpyHostToDevice);
  {
    GpuHalo halo(pr.d, o.gpu_mpi == "staged");
    s.st = gpu_cg(pr.g, pr.d, halo, rhs.p, x.p, cg_options(o));
  }
  s.x_host.assign(b.size, 0.0);
  s.t_d2h = timed_copy(s.x_host.data(), x.p, x.bytes, cudaMemcpyDeviceToHost);
  double tr[2] = {s.t_h2d, s.t_d2h};
  MPI_Allreduce(MPI_IN_PLACE, tr, 2, MPI_DOUBLE, MPI_MAX, pr.d.cart);
  s.t_h2d = tr[0];
  s.t_d2h = tr[1];
  s.err = max_error(pr.g, b, s.x_host.data());
  MPI_Allreduce(MPI_IN_PLACE, &s.err, 1, MPI_DOUBLE, MPI_MAX, pr.d.cart);
  // Same convention as the CPU build: total = slowest rank, phases = averages over ranks.
  MPI_Allreduce(MPI_IN_PLACE, &s.st.t_total, 1, MPI_DOUBLE, MPI_MAX, pr.d.cart);
  double ph[3] = {s.st.t_comp, s.st.t_halo, s.st.t_reduce};
  MPI_Allreduce(MPI_IN_PLACE, ph, 3, MPI_DOUBLE, MPI_SUM, pr.d.cart);
  s.st.t_comp = ph[0] / pr.d.size;
  s.st.t_halo = ph[1] / pr.d.size;
  s.st.t_reduce = ph[2] / pr.d.size;
  return s;
}

int run(const Options& o) {
  Problem pr = setup(o, o.Nx(), o.Ny(), o.Nz());
  const Solve s = gpu_solve(o, pr);
  if (pr.d.rank == 0) {
    Result r;
    r.machine = machine_name(o);
    r.backend = "mpi+cuda";
    r.dim = pr.g.dim;
    r.nx = pr.g.N[0];
    r.ny = pr.g.N[1];
    r.nz = pr.g.N[2];
    r.n = pr.g.n();
    r.nnz = pr.g.nnz();
    r.p = pr.d.size;
    r.threads = 1;
    r.gpus = pr.d.size;
    r.px = pr.d.dims[0];
    r.py = pr.d.dims[1];
    r.pz = pr.d.dims[2];
    r.decomp = o.decomp;
    r.halo = o.gpu_mpi;
    r.cg = "standard";
    r.iters = s.st.iters;
    r.rel_res = s.st.rel_res;
    r.err_inf = s.err;
    r.t_total = s.st.t_total;
    r.t_comp = s.st.t_comp;
    r.t_halo = s.st.t_halo;
    r.t_reduce = s.st.t_reduce;
    r.warmup = o.fixed_mode() ? o.warmup : 0;
    r.tol = o.tol;
    r.tag = o.tag;
    r.dot = "fast";
    r.t_h2d = s.t_h2d;
    r.t_d2h = s.t_d2h;
    if (!o.quiet) print_result(r);
    if (!o.csv.empty()) append_csv(o.csv, r);
  }
  free_decomp(pr.d);
  return 0;
}

// V3 / V2a on the GPU: A applied once on the device, compared with the CPU serial stencil.
int test_apply(const Options& o) {
  Problem pr = setup(o, o.Nx(), o.Ny(), o.Nz());
  const Block& b = pr.d.block;
  std::vector<double> v_h(b.size, 0.0), q_h(b.size, 0.0);
  fill_test_vector(pr.g, b, v_h.data());
  {
    DeviceVec v(b), q(b);
    CUDA_CHECK(cudaMemcpy(v.p, v_h.data(), v.bytes, cudaMemcpyHostToDevice));
    GpuHalo halo(pr.d, o.gpu_mpi == "staged");
    halo.exchange(v.p);
    gpu_apply_A(dev_block(b), dev_coef(pr.g), v.p, q.p);
    gpu_check("apply_A");
    CUDA_CHECK(cudaMemcpy(q_h.data(), q.p, q.bytes, cudaMemcpyDeviceToHost));
  }
  const std::vector<double> got = gather_to_root(pr.g, pr.d, q_h.data());
  int pass = 1;
  if (pr.d.rank == 0) {
    const Block gb = serial_block(pr.g);
    double* vs = alloc_vector(gb);
    double* qs = alloc_vector(gb);
    fill_test_vector(pr.g, gb, vs);
    apply_A(pr.g, gb, vs, qs, Region::All);
    long long mismatches = 0;
    double maxdiff = 0.0, maxval = 0.0;
    for (int i = 1; i <= gb.n[0]; ++i)
      for (int j = 1; j <= gb.n[1]; ++j)
        for (int k = gb.k0; k <= gb.k1; ++k) {
          const long id = gb.idx(i, j, k);
          maxval = std::fmax(maxval, std::fabs(qs[id]));
          if (got[id] != qs[id]) {
            ++mismatches;
            maxdiff = std::fmax(maxdiff, std::fabs(got[id] - qs[id]));
          }
        }
    pass = (maxdiff <= 1e-13 * maxval);
    std::printf("V3 apply (GPU) %dD %dx%dx%d p=%d [%dx%dx%d] gpu-mpi=%s: %lld of %lld values "
                "differ from the CPU (max |diff| %.3e, max |q| %.3e) -> %s%s\n",
                pr.g.dim, pr.g.N[0], pr.g.N[1], pr.g.N[2], pr.d.size, pr.d.dims[0],
                pr.d.dims[1], pr.d.dims[2], o.gpu_mpi.c_str(), mismatches, pr.g.n(), maxdiff,
                maxval, pass ? "PASS" : "FAIL", mismatches == 0 ? " (bit-identical)" : "");
    free_vector(vs);
    free_vector(qs);
  }
  MPI_Bcast(&pass, 1, MPI_INT, 0, pr.d.cart);
  free_decomp(pr.d);
  return pass ? 0 : 1;
}

// V3 / V2b on the GPU: a full solve compared with rank 0's serial CPU solve.
int test_solve(const Options& o_in) {
  Options o = o_in;
  if (o.fixed_mode()) {
    o.tol = 1e-10;
    o.iters = 100000;
  }
  Problem pr = setup(o, o.Nx(), o.Ny(), o.Nz());
  const Solve s = gpu_solve(o, pr);
  const std::vector<double> xp = gather_to_root(pr.g, pr.d, s.x_host.data());
  int pass = 1;
  if (pr.d.rank == 0) {
    const Block gb = serial_block(pr.g);
    double* rhs = alloc_vector(gb);
    double* xs = alloc_vector(gb);
    fill_rhs(pr.g, gb, rhs);
    SerialComm sc;
    const CGStats ss = cg_solve(pr.g, gb, sc, rhs, xs, cg_options(o));
    double num = 0.0, den = 0.0;
    for (int i = 1; i <= gb.n[0]; ++i)
      for (int j = 1; j <= gb.n[1]; ++j)
        for (int k = gb.k0; k <= gb.k1; ++k) {
          const long id = gb.idx(i, j, k);
          num += (xp[id] - xs[id]) * (xp[id] - xs[id]);
          den += xs[id] * xs[id];
        }
    const double rel = std::sqrt(num / den);
    pass = (std::abs(s.st.iters - ss.iters) <= 1) && rel <= o.check_tol;
    std::printf("V3 solve (GPU) %dD %dx%dx%d p=%d [%dx%dx%d] gpu-mpi=%s: iters %d vs CPU serial "
                "%d, ||x_gpu - x_cpu||/||x_cpu|| = %.3e, err_inf %.4e -> %s\n",
                pr.g.dim, pr.g.N[0], pr.g.N[1], pr.g.N[2], pr.d.size, pr.d.dims[0],
                pr.d.dims[1], pr.d.dims[2], o.gpu_mpi.c_str(), s.st.iters, ss.iters, rel, s.err,
                pass ? "PASS" : "FAIL");
    free_vector(rhs);
    free_vector(xs);
  }
  MPI_Bcast(&pass, 1, MPI_INT, 0, pr.d.cart);
  free_decomp(pr.d);
  return pass ? 0 : 1;
}

// V1 on the GPU: N + 1 = 16, 32, ...; the observed order should approach 2.
int test_mms(const Options& o_in) {
  Options o = o_in;
  if (o.fixed_mode() || o.tol > 1e-10) o.tol = 1e-10;
  o.iters = 100000;
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
    std::printf("V1 (manufactured solution), GPU p=%d, %dD, tol=%.0e\n%8s %12s %8s %12s %12s %8s\n",
                size, o.dim, o.tol, "N", "h", "iters", "rel_res", "err_inf", "order");
  double prev_err = 0.0, prev_h = 0.0, order = 0.0;
  for (int np1 = 16; np1 - 1 <= o.max_n; np1 *= 2) {
    const int N = np1 - 1;
    Problem pr = setup(o, N, N, N);
    const Solve s = gpu_solve(o, pr);
    order = (prev_err > 0) ? std::log(prev_err / s.err) / std::log(prev_h / pr.g.h[0]) : 0.0;
    if (rank == 0)
      std::printf("%8d %12.4e %8d %12.3e %12.4e %8.3f\n", N, pr.g.h[0], s.st.iters,
                  s.st.rel_res, s.err, order);
    prev_err = s.err;
    prev_h = pr.g.h[0];
    free_decomp(pr.d);
  }
  const bool pass = std::fabs(order - 2.0) < 0.1;
  if (rank == 0)
    std::printf("V1 %dD (GPU p=%d): observed order %.3f -> %s\n", o.dim, size, order,
                pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const int env_local = local_rank_from_env();
  int dev = (env_local >= 0) ? select_device(env_local) : -1;
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (dev < 0) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int local = 0;
    MPI_Comm_rank(node, &local);
    MPI_Comm_free(&node);
    dev = select_device(local);
  }
  const Options o = parse_options(argc, argv, "poisson_cuda", rank == 0);
  if (o.cg != "standard" && rank == 0)
    std::fprintf(stderr, "note: the CUDA build implements standard CG only; --cg ignored\n");
  if (o.dot != "fast" && rank == 0)
    std::fprintf(stderr, "note: --dot repro is implemented in the CPU builds only; ignored\n");

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const int aware = cuda_aware_at_runtime();
  if (rank == 0 && !o.quiet)
    std::printf("GPU: %s (%.0f GB, sm_%d%d), %d rank(s); MPI CUDA-aware at run time: %s\n",
                prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor, size,
                aware == 1 ? "yes" : aware == 0 ? "no" : "unknown");
  if (o.gpu_mpi == "aware" && aware == 0 && size > 1) {
    if (rank == 0)
      std::fprintf(stderr, "error: this MPI is not CUDA-aware; use --gpu-mpi staged\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  int rc = 0;
  if (o.test == "apply")
    rc = test_apply(o);
  else if (o.test == "solve")
    rc = test_solve(o);
  else if (o.test == "mms")
    rc = test_mms(o);
  else
    rc = run(o);
  MPI_Finalize();
  return rc;
}
