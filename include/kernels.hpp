// CPU kernels: the matrix-free stencil and the CG vector operations.
// All loops run over interior points only and are OpenMP-parallel when compiled with OpenMP
// (static schedule everywhere, so first-touch page placement matches the compute loops).
#pragma once
#include "dd.hpp"
#include "grid.hpp"

// Which interior points apply_A updates. Interior = points at least one cell away from every
// face of the block (their stencil never reads a ghost cell); Boundary = all the others.
// Interior + Boundary = All, each point exactly once (used by the overlap halo variant).
enum class Region { All, Interior, Boundary };

// q = A u. Every point is computed by the same expression in the same order, so results are
// bitwise identical however the grid is split (validation test V2a).
void apply_A(const Grid& g, const Block& b, const double* u, double* q, Region region);

double dot(const Block& b, const double* x, const double* y);
void dot2(const Block& b, const double* r, const double* w, double out[2]);  // {r.r, w.r}

// Standard CG:  x += alpha p;  r -= alpha q;  returns the local r.r of the new r.
double update_xr(const Block& b, double alpha, const double* p, const double* q, double* x,
                 double* r);
void update_p(const Block& b, double beta, const double* r, double* p);  // p = r + beta p

// Reproducible versions (--dot repro): the same results, accumulated in double-double so they
// do not depend on how the rows are split over ranks and threads (dd.hpp).
DD dot_dd(const Block& b, const double* x, const double* y);
void dot2_dd(const Block& b, const double* r, const double* w, DD out[2]);  // {r.r, w.r}
DD update_xr_dd(const Block& b, double alpha, const double* p, const double* q, double* x,
                double* r);

// Chronopoulos-Gear CG:  p = r + beta p;  s = w + beta s;  x += alpha p;  r -= alpha s.
void update_cg_fused(const Block& b, double alpha, double beta, const double* w, double* r,
                     double* p, double* s, double* x);

void copy_interior(const Block& b, const double* src, double* dst);
void zero_interior(const Block& b, double* v);

// 64-byte aligned vector, zeroed in parallel (first touch); release with free_vector.
double* alloc_vector(const Block& b);
void free_vector(double* v);
