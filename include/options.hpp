// Command-line options and result output (stdout summary + one CSV row per run),
// shared by the serial, MPI and MPI+CUDA drivers.
#pragma once
#include <string>

struct Options {
  int dim = 3;
  int n = 64;                  // --n: interior points per dimension
  int nx = 0, ny = 0, nz = 0;  // --nx/--ny/--nz override --n per dimension
  int nlocal = 0;              // --nlocal L (MPI): global N_k = L * dims_k (weak scaling)
  int iters = 0;               // --iters K
  double tol = -1.0;           // --tol T (< 0: not given)
  int warmup = 10;             // --warmup W
  std::string decomp = "block";      // block | pencil | slab
  std::string halo = "nonblocking";  // sendrecv | nonblocking | overlap | rma | pscw  (CPU)
  std::string dot = "fast";          // fast | repro (double-double, bit-identical for any p)
  std::string gpu_mpi = "aware";     // aware | staged  (CUDA build)
  std::string cg = "standard";       // standard | fused
  std::string csv;                   // append a CSV row to this file
  std::string machine;               // machine name recorded in the CSV
  std::string tag;                   // experiment label recorded in the CSV (e.g. strong)
  std::string test;                  // mms | apply | solve  (validation modes)
  double check_tol = 1e-10;          // --check-tol: pass threshold of the solve test
  int max_n = 255;                   // --max-n: largest N of the mms test
  bool quiet = false;

  int Nx() const { return nx > 0 ? nx : n; }
  int Ny() const { return ny > 0 ? ny : n; }
  int Nz() const { return dim == 3 ? (nz > 0 ? nz : n) : 1; }
  bool fixed_mode() const { return !(tol > 0); }
};

// Parses argv; prints usage and exits on --help or an error. Applies the defaults:
//   only --iters K     -> fixed mode, K timed iterations after --warmup W
//   --tol T (> 0)      -> run to ||r||/||b|| < T, at most --iters (default 100000)
//   neither            -> --tol 1e-8
Options parse_options(int argc, char** argv, const char* program, bool print_errors);

// --machine, else $CG_MACHINE, else the host name.
std::string machine_name(const Options& o);

struct Result {
  std::string machine, backend;
  int dim = 3, nx = 1, ny = 1, nz = 1;
  long long n = 0, nnz = 0;
  int p = 1, threads = 1, gpus = 0;
  std::string decomp, halo, cg;
  int iters = 0;
  double rel_res = 0, err_inf = 0;
  double t_total = 0, t_comp = 0, t_halo = 0, t_reduce = 0;
  int px = 1, py = 1, pz = 1;
  int warmup = 0;
  double tol = 0;
  std::string tag;
  std::string dot = "fast";
  double t_h2d = 0, t_d2h = 0;  // CUDA build: one-off host->device / device->host copies
};

void print_result(const Result& r);
void append_csv(const std::string& path, const Result& r);  // writes the header on first use
