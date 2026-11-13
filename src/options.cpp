#include "options.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void usage(const char* program) {
  std::printf(
      "usage: %s [options]\n"
      "  --dim 2|3                 problem dimension (default 3)\n"
      "  --n N | --nx --ny --nz    interior points per dimension (default 64)\n"
      "  --nlocal L                MPI weak scaling: global N_k = L x (ranks along k)\n"
      "  --iters K                 alone: fixed mode, K timed iterations after the warm-up\n"
      "                            with --tol: maximum iterations\n"
      "  --tol T                   stop when ||r||/||b|| < T (default 1e-8 if --iters absent)\n"
      "  --warmup W                untimed iterations in fixed mode (default 10)\n"
      "  --decomp block|pencil|slab   MPI decomposition (default block)\n"
      "  --halo sendrecv|nonblocking|overlap|rma|pscw   MPI halo exchange (default nonblocking)\n"
      "  --dot fast|repro          repro: double-double dot products, bit-identical results\n"
      "                            for any number of ranks and threads (CPU builds)\n"
      "  --gpu-mpi aware|staged    CUDA build: CUDA-aware or host-staged halo (default aware)\n"
      "  --cg standard|fused       standard CG or Chronopoulos-Gear (default standard)\n"
      "  --csv FILE                append one result row to FILE\n"
      "  --machine NAME            machine name for the CSV (default $CG_MACHINE or host)\n"
      "  --tag LABEL               experiment label for the CSV (e.g. strong, weak)\n"
      "  --test mms|apply|solve    validation: V1 convergence order | V2a bitwise A*x\n"
      "                            against serial | V2b CG solution against serial\n"
      "  --max-n N                 largest N in the mms test (default 255)\n"
      "  --check-tol E             pass threshold of the solve test (default 1e-10)\n"
      "  --quiet                   no summary on stdout\n",
      program);
}

[[noreturn]] void fail(const char* program, const std::string& msg, bool print) {
  if (print) {
    std::fprintf(stderr, "%s: %s\n", program, msg.c_str());
    std::fprintf(stderr, "run %s --help for the options\n", program);
  }
  std::exit(2);
}

bool one_of(const std::string& v, std::initializer_list<const char*> allowed) {
  for (const char* a : allowed)
    if (v == a) return true;
  return false;
}

}  // namespace

Options parse_options(int argc, char** argv, const char* program, bool print) {
  Options o;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) fail(program, std::string("missing value after ") + argv[i], print);
    return argv[++i];
  };
  auto to_int = [&](const std::string& s, const char* what) {
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || v < 0 || v > 1000000000L)
      fail(program, std::string("bad value for ") + what + ": " + s, print);
    return (int)v;
  };
  auto to_double = [&](const std::string& s, const char* what) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0' || v < 0)
      fail(program, std::string("bad value for ") + what + ": " + s, print);
    return v;
  };
  bool iters_given = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      if (print) usage(program);
      std::exit(0);
    } else if (a == "--dim") o.dim = to_int(need(i), "--dim");
    else if (a == "--n") o.n = to_int(need(i), "--n");
    else if (a == "--nx") o.nx = to_int(need(i), "--nx");
    else if (a == "--ny") o.ny = to_int(need(i), "--ny");
    else if (a == "--nz") o.nz = to_int(need(i), "--nz");
    else if (a == "--nlocal") o.nlocal = to_int(need(i), "--nlocal");
    else if (a == "--iters") { o.iters = to_int(need(i), "--iters"); iters_given = true; }
    else if (a == "--tol") o.tol = to_double(need(i), "--tol");
    else if (a == "--warmup") o.warmup = to_int(need(i), "--warmup");
    else if (a == "--decomp") o.decomp = need(i);
    else if (a == "--halo") o.halo = need(i);
    else if (a == "--dot") o.dot = need(i);
    else if (a == "--gpu-mpi") o.gpu_mpi = need(i);
    else if (a == "--cg") o.cg = need(i);
    else if (a == "--csv") o.csv = need(i);
    else if (a == "--machine") o.machine = need(i);
    else if (a == "--tag") o.tag = need(i);
    else if (a == "--test") o.test = need(i);
    else if (a == "--check-tol") o.check_tol = to_double(need(i), "--check-tol");
    else if (a == "--max-n") o.max_n = to_int(need(i), "--max-n");
    else if (a == "--quiet") o.quiet = true;
    else fail(program, "unknown option " + a, print);
  }
  if (o.dim != 2 && o.dim != 3) fail(program, "--dim must be 2 or 3", print);
  if (o.Nx() < 1 || o.Ny() < 1 || o.Nz() < 1) fail(program, "grid sizes must be >= 1", print);
  if (!one_of(o.decomp, {"block", "pencil", "slab"}))
    fail(program, "--decomp must be block, pencil or slab", print);
  if (!one_of(o.halo, {"sendrecv", "nonblocking", "overlap", "rma", "pscw"}))
    fail(program, "--halo must be sendrecv, nonblocking, overlap, rma or pscw", print);
  if (!one_of(o.dot, {"fast", "repro"})) fail(program, "--dot must be fast or repro", print);
  if (!one_of(o.gpu_mpi, {"aware", "staged"}))
    fail(program, "--gpu-mpi must be aware or staged", print);
  if (!one_of(o.cg, {"standard", "fused"})) fail(program, "--cg must be standard or fused", print);
  if (!o.test.empty() && !one_of(o.test, {"mms", "apply", "solve"}))
    fail(program, "--test must be mms, apply or solve", print);
  if (o.tol < 0) o.tol = iters_given ? 0.0 : 1e-8;
  if (o.tol > 0 && !iters_given) o.iters = 100000;
  if (o.fixed_mode() && o.iters < 1) fail(program, "fixed mode needs --iters >= 1", print);
  return o;
}

std::string machine_name(const Options& o) {
  if (!o.machine.empty()) return o.machine;
  if (const char* m = std::getenv("CG_MACHINE")) return m;
  char host[256] = {0};
  if (gethostname(host, sizeof host - 1) == 0) {
    std::string h = host;
    const auto dot = h.find('.');
    return dot == std::string::npos ? h : h.substr(0, dot);
  }
  return "unknown";
}

void print_result(const Result& r) {
  const double per = r.iters > 0 ? r.t_total / r.iters : 0.0;
  std::printf(
      "%s %dD %dx%dx%d (n=%lld, nnz=%lld) p=%d [%dx%dx%d] threads=%d gpus=%d decomp=%s "
      "halo=%s cg=%s dot=%s\n"
      "  iters=%d rel_res=%.3e err_inf=%.3e\n"
      "  t_total=%.4f s (%.3f ms/iter)  comp=%.4f  halo=%.4f  reduce=%.4f\n",
      r.backend.c_str(), r.dim, r.nx, r.ny, r.nz, r.n, r.nnz, r.p, r.px, r.py, r.pz, r.threads,
      r.gpus, r.decomp.c_str(), r.halo.c_str(), r.cg.c_str(), r.dot.c_str(), r.iters,
      r.rel_res, r.err_inf, r.t_total, 1e3 * per, r.t_comp, r.t_halo, r.t_reduce);
  if (r.t_h2d > 0 || r.t_d2h > 0)
    std::printf("  one-off transfers: host->device %.4f s, device->host %.4f s (= %.1f iterations)\n",
                r.t_h2d, r.t_d2h, per > 0 ? (r.t_h2d + r.t_d2h) / per : 0.0);
  std::fflush(stdout);
}

void append_csv(const std::string& path, const Result& r) {
  std::FILE* f = std::fopen(path.c_str(), "a");
  if (!f) {
    std::fprintf(stderr, "cannot open %s for appending\n", path.c_str());
    return;
  }
  std::fseek(f, 0, SEEK_END);
  if (std::ftell(f) == 0)
    std::fputs(
        "machine,backend,dim,nx,ny,nz,n,nnz,p,threads,gpus,decomp,halo,cg,iters,rel_res,"
        "err_inf,t_total,t_comp,t_halo,t_reduce,px,py,pz,warmup,tol,t_iter,tag,dot,t_h2d,"
        "t_d2h\n",
        f);
  const double per = r.iters > 0 ? r.t_total / r.iters : 0.0;
  char line[1024];
  std::snprintf(line, sizeof line,
                "%s,%s,%d,%d,%d,%d,%lld,%lld,%d,%d,%d,%s,%s,%s,%d,%.6e,%.6e,%.6e,%.6e,%.6e,"
                "%.6e,%d,%d,%d,%d,%.3e,%.6e,%s,%s,%.6e,%.6e\n",
                r.machine.c_str(), r.backend.c_str(), r.dim, r.nx, r.ny, r.nz, r.n, r.nnz, r.p,
                r.threads, r.gpus, r.decomp.c_str(), r.halo.c_str(), r.cg.c_str(), r.iters,
                r.rel_res, r.err_inf, r.t_total, r.t_comp, r.t_halo, r.t_reduce, r.px, r.py,
                r.pz, r.warmup, r.tol, per, r.tag.c_str(), r.dot.c_str(), r.t_h2d, r.t_d2h);
  std::fputs(line, f);  // one write per row, so concurrent jobs do not interleave rows
  std::fclose(f);
}
