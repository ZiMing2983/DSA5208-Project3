// Grid, local block layout and manufactured solution for
//     -lap(u) = f  on (0,1)^d,  u = 0 on the boundary,  d = 2 or 3,
// discretised with the 5-point (2D) / 7-point (3D) finite-difference stencil.
#pragma once
#include <cmath>
#include <cstddef>

// The global problem. N[k] interior points in dimension k, spacing h[k] = 1/(N[k]+1).
// In 2D the third dimension is inactive: N[2] = 1 and c[2] = 0.
struct Grid {
  int dim = 3;
  int N[3] = {1, 1, 1};
  double h[3] = {1.0, 1.0, 1.0};
  double c[3] = {0.0, 0.0, 0.0};  // 1/h^2 per active dimension
  double diag = 0.0;              // 2 * (c[0] + c[1] + c[2])
  long long n() const { return (long long)N[0] * N[1] * N[2]; }  // unknowns
  long long nnz() const;                                           // nonzeros of A
};

Grid make_grid(int dim, int nx, int ny, int nz);

// One rank's block of interior points, stored with a 1-cell ghost layer in every active
// dimension:  idx(i,j,k) = (i*sy + j)*sz + k,  k fastest.
//   interior: i in [1, n[0]], j in [1, n[1]], k in [k0, k1]
//   3D: sz = n[2]+2, k0 = 1, k1 = n[2].   2D: sz = 1 (no ghost layer in z), k0 = k1 = 0.
// Ghost cells on the physical boundary hold 0 (Dirichlet); the others are filled by the
// halo exchange.
struct Block {
  int dim = 3;
  int n[3] = {1, 1, 1};    // interior extents (n[2] = 1 in 2D)
  int off[3] = {0, 0, 0};  // global 0-based index of the first interior point
  int sx = 1, sy = 1, sz = 1;
  long si = 1, sj = 1;     // strides of i and j; the stride of k is 1
  int k0 = 0, k1 = 0;
  std::size_t size = 0;    // doubles per vector, ghosts included
  long idx(int i, int j, int k) const { return (long)i * si + (long)j * sj + k; }
  long long interior() const { return (long long)n[0] * n[1] * n[2]; }
};

Block make_block(const Grid& g, const int n[3], const int off[3]);
Block serial_block(const Grid& g);

// Manufactured solution u_ex = prod_k g(x_k) with g(t) = t(1-t)e^t, so that
// f = sum_k q(x_k) prod_{j != k} g(x_j) with q(t) = -g''(t) = t(t+3)e^t.
// (Not prod sin(pi x_k): that is a discrete eigenvector, and CG would converge in one step.)
inline double mms_g(double t) { return t * (1.0 - t) * std::exp(t); }
inline double mms_q(double t) { return t * (t + 3.0) * std::exp(t); }

void fill_rhs(const Grid& g, const Block& b, double* f);           // interior values of f
double max_error(const Grid& g, const Block& b, const double* x);  // local max |x - u_ex|

// A non-smooth, deterministic test vector defined by global indices, so every rank produces
// exactly the values the serial code produces (validation test V2a).
void fill_test_vector(const Grid& g, const Block& b, double* v);
