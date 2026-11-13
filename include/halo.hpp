// MPI domain decomposition and halo exchange.
#pragma once
#include <mpi.h>

#include <string>
#include <vector>

#include "cg.hpp"
#include "grid.hpp"

// Cartesian block decomposition (non-periodic). Rank coordinates c_k in [0, dims_k); each rank
// owns floor(N_k/P_k) points along k, and the first N_k mod P_k ranks one more:
//     offset o_k = c_k * floor(N_k/P_k) + min(c_k, N_k mod P_k).
struct Decomp {
  MPI_Comm cart = MPI_COMM_NULL;
  int size = 1, rank = 0;
  int dims[3] = {1, 1, 1};
  int coords[3] = {0, 0, 0};
  int nbr[3][2] = {{MPI_PROC_NULL, MPI_PROC_NULL},   // [dimension][0 = lower, 1 = upper]
                   {MPI_PROC_NULL, MPI_PROC_NULL},   // MPI_PROC_NULL on the physical boundary
                   {MPI_PROC_NULL, MPI_PROC_NULL}};
  Block block;
};

// Process-grid shape for p ranks: block = MPI_Dims_create over all dimensions,
// pencil = no split along z (3D), slab = split along x only. dims[2] = 1 in 2D.
void decomp_dims(int p, int dim, const std::string& kind, int dims[3]);

// Creates the Cartesian communicator and this rank's block. Aborts (all ranks) if some rank
// would own no points.
Decomp make_decomp(const Grid& g, MPI_Comm world, const int dims[3]);
void free_decomp(Decomp& d);

// Face buffers: dimension d, interior layer `layer` (1 or n[d]) or ghost layer (0 or n[d]+1).
long face_size(const Block& b, int d);
void pack_face(const Block& b, int d, int layer, const double* u, double* buf);
void unpack_face(const Block& b, int d, int layer, const double* buf, double* u);

enum class HaloKind { Sendrecv, Nonblocking, Overlap, Rma, Pscw };
HaloKind parse_halo(const std::string& s);

// CPU halo exchange + Allreduce. Variants (--halo):
//   sendrecv     blocking MPI_Sendrecv, one dimension after the other
//   nonblocking  MPI_Irecv/MPI_Isend on all faces, then MPI_Waitall
//   overlap      nonblocking, with interior points computed while messages are in flight
//   rma          one-sided MPI_Put into the neighbours' receive window, MPI_Win_fence epochs
//                (a fence synchronises every rank of the communicator)
//   pscw         the same Puts, synchronised only with the neighbours through
//                MPI_Win_post/start/complete/wait on an MPI group of the neighbouring ranks
// allreduce_dd: reproducible sum of double-double values (custom MPI datatype and MPI_Op).
class MpiComm final : public Comm {
 public:
  MpiComm(const Decomp& d, HaloKind kind);
  ~MpiComm() override;
  MpiComm(const MpiComm&) = delete;
  MpiComm& operator=(const MpiComm&) = delete;

  void halo_begin(double* u) override;
  void halo_end(double* u) override;
  bool overlap() const override { return kind_ == HaloKind::Overlap; }
  void allreduce_sum(double* v, int n) override;
  void allreduce_dd(DD* v, int n) override;

 private:
  bool has(int d, int s) const { return d_.nbr[d][s] != MPI_PROC_NULL; }
  int inner(int d, int s) const { return s == 0 ? 1 : d_.block.n[d]; }
  int ghost(int d, int s) const { return s == 0 ? 0 : d_.block.n[d] + 1; }
  void sendrecv_all(double* u);

  const Decomp& d_;
  HaloKind kind_;
  std::vector<double> send_[3][2], recv_[3][2];
  MPI_Request req_[12];
  int nreq_ = 0;
  // RMA: one window over all receive slots; remote_[d][s] = slot offset in neighbour's window.
  double* win_buf_ = nullptr;
  MPI_Win win_ = MPI_WIN_NULL;
  MPI_Aint slot_[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  MPI_Aint remote_[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  MPI_Group nbr_group_ = MPI_GROUP_NULL;  // PSCW: the neighbouring ranks
  MPI_Datatype dd_type_ = MPI_DATATYPE_NULL;
  MPI_Op dd_op_ = MPI_OP_NULL;
};

// Gathers the interior of a distributed vector into a serial-layout vector on rank 0
// (empty on the other ranks). Used by the validation tests.
std::vector<double> gather_to_root(const Grid& g, const Decomp& d, const double* v);
