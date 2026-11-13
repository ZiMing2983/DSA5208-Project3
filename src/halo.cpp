#include "halo.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

void decomp_dims(int p, int dim, const std::string& kind, int dims[3]) {
  dims[0] = dims[1] = dims[2] = 0;
  if (kind == "slab") {
    dims[1] = dims[2] = 1;
  } else if (kind == "pencil" && dim == 3) {
    dims[2] = 1;
  }
  MPI_Dims_create(p, dim, dims);
  if (dim == 2) dims[2] = 1;
}

Decomp make_decomp(const Grid& g, MPI_Comm world, const int dims_in[3]) {
  Decomp d;
  int periods[3] = {0, 0, 0};
  int dims[3] = {dims_in[0], dims_in[1], g.dim == 3 ? dims_in[2] : 1};
  MPI_Cart_create(world, g.dim, dims, periods, 0, &d.cart);
  MPI_Comm_size(d.cart, &d.size);
  MPI_Comm_rank(d.cart, &d.rank);
  int coords[3] = {0, 0, 0};
  MPI_Cart_coords(d.cart, d.rank, g.dim, coords);
  int n[3] = {1, 1, 1}, off[3] = {0, 0, 0};
  int ok = 1;
  for (int k = 0; k < 3; ++k) {
    d.dims[k] = dims[k];
    d.coords[k] = coords[k];
    if (k >= g.dim) continue;
    MPI_Cart_shift(d.cart, k, 1, &d.nbr[k][0], &d.nbr[k][1]);
    const int base = g.N[k] / dims[k], extra = g.N[k] % dims[k];
    n[k] = base + (coords[k] < extra ? 1 : 0);
    off[k] = coords[k] * base + std::min(coords[k], extra);
    if (n[k] < 1) ok = 0;
  }
  MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, d.cart);
  if (!ok) {
    if (d.rank == 0)
      std::fprintf(stderr, "error: %d x %d x %d ranks is more than the %d x %d x %d grid allows\n",
                   dims[0], dims[1], dims[2], g.N[0], g.N[1], g.N[2]);
    MPI_Abort(d.cart, 2);
  }
  d.block = make_block(g, n, off);
  return d;
}

void free_decomp(Decomp& d) {
  if (d.cart != MPI_COMM_NULL) MPI_Comm_free(&d.cart);
}

long face_size(const Block& b, int d) {
  const long nk = b.k1 - b.k0 + 1;
  if (d == 0) return (long)b.n[1] * nk;
  if (d == 1) return (long)b.n[0] * nk;
  return (long)b.n[0] * b.n[1];
}

// Faces are walked in (outer, inner) = (j,k), (i,k) or (i,j) order on both sides, so the
// sender's packing order matches the receiver's unpacking order.
void pack_face(const Block& b, int d, int layer, const double* u, double* buf) {
  const int k0 = b.k0, k1 = b.k1, nk = k1 - k0 + 1;
  [[maybe_unused]] const bool big = face_size(b, d) > 32768;  // only used by OpenMP
  if (d == 0) {
#pragma omp parallel for schedule(static) if (big)
    for (int j = 1; j <= b.n[1]; ++j) {
      const double* src = u + b.idx(layer, j, 0);
      double* dst = buf + (long)(j - 1) * nk - k0;
      for (int k = k0; k <= k1; ++k) dst[k] = src[k];
    }
  } else if (d == 1) {
#pragma omp parallel for schedule(static) if (big)
    for (int i = 1; i <= b.n[0]; ++i) {
      const double* src = u + b.idx(i, layer, 0);
      double* dst = buf + (long)(i - 1) * nk - k0;
      for (int k = k0; k <= k1; ++k) dst[k] = src[k];
    }
  } else {
#pragma omp parallel for schedule(static) if (big)
    for (int i = 1; i <= b.n[0]; ++i) {
      double* dst = buf + (long)(i - 1) * b.n[1];
      for (int j = 1; j <= b.n[1]; ++j) dst[j - 1] = u[b.idx(i, j, layer)];
    }
  }
}

void unpack_face(const Block& b, int d, int layer, const double* buf, double* u) {
  const int k0 = b.k0, k1 = b.k1, nk = k1 - k0 + 1;
  [[maybe_unused]] const bool big = face_size(b, d) > 32768;  // only used by OpenMP
  if (d == 0) {
#pragma omp parallel for schedule(static) if (big)
    for (int j = 1; j <= b.n[1]; ++j) {
      double* dst = u + b.idx(layer, j, 0);
      const double* src = buf + (long)(j - 1) * nk - k0;
      for (int k = k0; k <= k1; ++k) dst[k] = src[k];
    }
  } else if (d == 1) {
#pragma omp parallel for schedule(static) if (big)
    for (int i = 1; i <= b.n[0]; ++i) {
      double* dst = u + b.idx(i, layer, 0);
      const double* src = buf + (long)(i - 1) * nk - k0;
      for (int k = k0; k <= k1; ++k) dst[k] = src[k];
    }
  } else {
#pragma omp parallel for schedule(static) if (big)
    for (int i = 1; i <= b.n[0]; ++i) {
      const double* src = buf + (long)(i - 1) * b.n[1];
      for (int j = 1; j <= b.n[1]; ++j) u[b.idx(i, j, layer)] = src[j - 1];
    }
  }
}

HaloKind parse_halo(const std::string& s) {
  if (s == "sendrecv") return HaloKind::Sendrecv;
  if (s == "overlap") return HaloKind::Overlap;
  if (s == "rma") return HaloKind::Rma;
  if (s == "pscw") return HaloKind::Pscw;
  return HaloKind::Nonblocking;
}

// Custom MPI_Op for the reproducible Allreduce (Lecture 8: custom operations): element-wise
// double-double addition. Declared commutative; dd_add is (to ~1e-32) associative too.
static void dd_sum_op(void* in, void* inout, int* len, MPI_Datatype*) {
  const DD* a = static_cast<const DD*>(in);
  DD* b = static_cast<DD*>(inout);
  for (int i = 0; i < *len; ++i) b[i] = dd_add(a[i], b[i]);
}

// Message tags: a face sent towards side s of dimension d has tag 2d + s; the receiver, which
// gets it on its opposite side, expects the same tag.
static int tag(int d, int s) { return 2 * d + s; }

MpiComm::MpiComm(const Decomp& d, HaloKind kind) : d_(d), kind_(kind) {
  const Block& b = d.block;
  const bool one_sided = (kind_ == HaloKind::Rma || kind_ == HaloKind::Pscw);
  for (int k = 0; k < b.dim; ++k)
    for (int s = 0; s < 2; ++s) {
      send_[k][s].assign(face_size(b, k), 0.0);
      if (!one_sided) recv_[k][s].assign(face_size(b, k), 0.0);
    }
  MPI_Type_contiguous(2, MPI_DOUBLE, &dd_type_);
  MPI_Type_commit(&dd_type_);
  MPI_Op_create(&dd_sum_op, 1, &dd_op_);
  if (kind_ == HaloKind::Pscw) {
    // PSCW synchronises only with the neighbours: an MPI group of the neighbouring ranks
    // (Lecture 8: groups) is both the exposure group (who may Put into my window) and the
    // access group (whose windows I Put into).
    int ranks[6], n = 0;
    for (int k = 0; k < b.dim; ++k)
      for (int s = 0; s < 2; ++s)
        if (d.nbr[k][s] != MPI_PROC_NULL) {
          bool seen = false;
          for (int i = 0; i < n; ++i) seen = seen || (ranks[i] == d.nbr[k][s]);
          if (!seen) ranks[n++] = d.nbr[k][s];
        }
    MPI_Group cart_group;
    MPI_Comm_group(d.cart, &cart_group);
    MPI_Group_incl(cart_group, n, ranks, &nbr_group_);
    MPI_Group_free(&cart_group);
  }
  if (one_sided) {
    MPI_Aint total = 0;
    for (int k = 0; k < b.dim; ++k)
      for (int s = 0; s < 2; ++s) {
        slot_[k][s] = total;
        total += face_size(b, k);
      }
    MPI_Win_allocate(std::max<MPI_Aint>(total, 1) * (MPI_Aint)sizeof(double), sizeof(double),
                     MPI_INFO_NULL, d.cart, &win_buf_, &win_);
    // Each neighbour Puts its face into my slot on its side; tell it where that slot is.
    for (int k = 0; k < b.dim; ++k)
      for (int s = 0; s < 2; ++s)
        MPI_Sendrecv(&slot_[k][s], 1, MPI_AINT, d.nbr[k][s], 100 + tag(k, s), &remote_[k][s], 1,
                     MPI_AINT, d.nbr[k][s], 100 + tag(k, 1 - s), d.cart, MPI_STATUS_IGNORE);
  }
}

MpiComm::~MpiComm() {
  if (win_ != MPI_WIN_NULL) MPI_Win_free(&win_);
  if (nbr_group_ != MPI_GROUP_NULL && nbr_group_ != MPI_GROUP_EMPTY) MPI_Group_free(&nbr_group_);
  if (dd_op_ != MPI_OP_NULL) MPI_Op_free(&dd_op_);
  if (dd_type_ != MPI_DATATYPE_NULL) MPI_Type_free(&dd_type_);
}

void MpiComm::sendrecv_all(double* u) {
  const Block& b = d_.block;
  for (int k = 0; k < b.dim; ++k) {
    const int m = (int)face_size(b, k);
    for (int s = 0; s < 2; ++s) {
      // Send my side-s layer to the side-s neighbour; receive my opposite ghost layer.
      const int dst = d_.nbr[k][s], src = d_.nbr[k][1 - s];
      if (dst != MPI_PROC_NULL) pack_face(b, k, inner(k, s), u, send_[k][s].data());
      MPI_Sendrecv(send_[k][s].data(), m, MPI_DOUBLE, dst, tag(k, s), recv_[k][1 - s].data(), m,
                   MPI_DOUBLE, src, tag(k, s), d_.cart, MPI_STATUS_IGNORE);
      if (src != MPI_PROC_NULL) unpack_face(b, k, ghost(k, 1 - s), recv_[k][1 - s].data(), u);
    }
  }
}

void MpiComm::halo_begin(double* u) {
  const Block& b = d_.block;
  switch (kind_) {
    case HaloKind::Sendrecv:
      sendrecv_all(u);
      break;
    case HaloKind::Nonblocking:
    case HaloKind::Overlap:
      nreq_ = 0;
      for (int k = 0; k < b.dim; ++k)
        for (int s = 0; s < 2; ++s)
          if (has(k, s))
            MPI_Irecv(recv_[k][s].data(), (int)face_size(b, k), MPI_DOUBLE, d_.nbr[k][s],
                      tag(k, 1 - s), d_.cart, &req_[nreq_++]);
      for (int k = 0; k < b.dim; ++k)
        for (int s = 0; s < 2; ++s)
          if (has(k, s)) {
            pack_face(b, k, inner(k, s), u, send_[k][s].data());
            MPI_Isend(send_[k][s].data(), (int)face_size(b, k), MPI_DOUBLE, d_.nbr[k][s],
                      tag(k, s), d_.cart, &req_[nreq_++]);
          }
      break;
    case HaloKind::Rma:
    case HaloKind::Pscw:
      for (int k = 0; k < b.dim; ++k)
        for (int s = 0; s < 2; ++s)
          if (has(k, s)) pack_face(b, k, inner(k, s), u, send_[k][s].data());
      if (kind_ == HaloKind::Rma) {
        MPI_Win_fence(0, win_);  // collective over every rank of the communicator
      } else {
        MPI_Win_post(nbr_group_, 0, win_);   // expose my window to my neighbours
        MPI_Win_start(nbr_group_, 0, win_);  // open access to my neighbours' windows
      }
      for (int k = 0; k < b.dim; ++k)
        for (int s = 0; s < 2; ++s)
          if (has(k, s)) {
            const int m = (int)face_size(b, k);
            MPI_Put(send_[k][s].data(), m, MPI_DOUBLE, d_.nbr[k][s], remote_[k][s], m,
                    MPI_DOUBLE, win_);
          }
      break;
  }
}

void MpiComm::halo_end(double* u) {
  const Block& b = d_.block;
  switch (kind_) {
    case HaloKind::Sendrecv:
      break;
    case HaloKind::Nonblocking:
    case HaloKind::Overlap:
      MPI_Waitall(nreq_, req_, MPI_STATUSES_IGNORE);
      for (int k = 0; k < b.dim; ++k)
        for (int s = 0; s < 2; ++s)
          if (has(k, s)) unpack_face(b, k, ghost(k, s), recv_[k][s].data(), u);
      break;
    case HaloKind::Rma:
    case HaloKind::Pscw:
      if (kind_ == HaloKind::Rma) {
        MPI_Win_fence(0, win_);
      } else {
        MPI_Win_complete(win_);  // my Puts are done
        MPI_Win_wait(win_);      // my neighbours' Puts into my window are done
      }
      for (int k = 0; k < b.dim; ++k)
        for (int s = 0; s < 2; ++s)
          if (has(k, s)) unpack_face(b, k, ghost(k, s), win_buf_ + slot_[k][s], u);
      break;
  }
}

void MpiComm::allreduce_sum(double* v, int n) {
  MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, d_.cart);
}

void MpiComm::allreduce_dd(DD* v, int n) {
  MPI_Allreduce(MPI_IN_PLACE, v, n, dd_type_, dd_op_, d_.cart);
}

std::vector<double> gather_to_root(const Grid& g, const Decomp& d, const double* v) {
  const Block& b = d.block;
  const long long cnt = b.interior();
  std::vector<double> mine;
  mine.reserve(cnt);
  for (int i = 1; i <= b.n[0]; ++i)
    for (int j = 1; j <= b.n[1]; ++j)
      for (int k = b.k0; k <= b.k1; ++k) mine.push_back(v[b.idx(i, j, k)]);
  int meta[6] = {b.off[0], b.off[1], b.off[2], b.n[0], b.n[1], b.n[2]};
  std::vector<int> all_meta(d.rank == 0 ? 6 * d.size : 0);
  MPI_Gather(meta, 6, MPI_INT, all_meta.data(), 6, MPI_INT, 0, d.cart);
  std::vector<int> counts, displs;
  std::vector<double> all;
  if (d.rank == 0) {
    counts.resize(d.size);
    displs.resize(d.size);
    long long total = 0;
    for (int r = 0; r < d.size; ++r) {
      const int* m = &all_meta[6 * r];
      counts[r] = m[3] * m[4] * m[5];
      displs[r] = (int)total;
      total += counts[r];
    }
    all.resize(total);
  }
  MPI_Gatherv(mine.data(), (int)cnt, MPI_DOUBLE, all.data(), counts.data(), displs.data(),
              MPI_DOUBLE, 0, d.cart);
  std::vector<double> out;
  if (d.rank == 0) {
    const Block gb = serial_block(g);
    out.assign(gb.size, 0.0);
    for (int r = 0; r < d.size; ++r) {
      const int* m = &all_meta[6 * r];
      const double* src = all.data() + displs[r];
      long t = 0;
      for (int i = 1; i <= m[3]; ++i)
        for (int j = 1; j <= m[4]; ++j)
          for (int k = b.k0; k <= (g.dim == 3 ? m[5] : 0); ++k)
            out[gb.idx(m[0] + i, m[1] + j, g.dim == 3 ? m[2] + k : 0)] = src[t++];
    }
  }
  return out;
}
