// Double-double ("DD") accumulation for reproducible dot products (--dot repro).
//
// Floating-point addition is not associative (Lecture 8), so an ordinary parallel sum changes
// in its last bits with the number of ranks and threads. A DD value hi + lo carries about 106
// bits: TwoSum keeps every rounding error exactly, so the final rounded value hi + lo is the
// correctly rounded sum except when the exact sum lies within ~1e-27 (relative) of a rounding
// boundary. In practice the result is the same for any order of the terms, i.e. for any
// decomposition and thread count. (Requires strict IEEE arithmetic: no -ffast-math, no FMA.)
#pragma once

struct DD {
  double hi = 0.0;
  double lo = 0.0;
};

// s + e == a + b exactly (Knuth's TwoSum; no assumption on |a| vs |b|).
inline void two_sum(double a, double b, double& s, double& e) {
  s = a + b;
  const double bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

// a += x, with the rounding error of the hi sum kept in lo.
inline void dd_acc(DD& a, double x) {
  double s, e;
  two_sum(a.hi, x, s, e);
  a.hi = s;
  a.lo += e;
}

// Sum of two DD values, renormalised. Commutative, as MPI_Op_create(..., commute=1) assumes.
inline DD dd_add(DD a, DD b) {
  double s, e;
  two_sum(a.hi, b.hi, s, e);
  e += a.lo + b.lo;
  DD r;
  r.hi = s + e;
  r.lo = e - (r.hi - s);
  return r;
}

inline double dd_value(DD a) { return a.hi + a.lo; }
