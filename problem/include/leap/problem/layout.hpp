#pragma once

// The decision vector and the row space.
//
//   y = [ x_elem0 | x_elem1 | ... | gamma ]     primal x (np entries) first, multipliers
//                                                gamma (md entries) after.
// Within an element the fields are stored dim x columns, column-major (node k = column k):
//   Q (nq x Nn), V (nv x Nn), A (na x Nn), C (nlam x Nc, the force control points),
//   U (nu x Nn, absent under torque elimination).
// The physical node force is lam_k = sum_j E(k, j) sigma_j C_j (E = Bernstein basis of
// degree Nc - 1 at the CGL nodes, sigma the column scaling), and the force at any tau is
// sum_j B_j((tau + 1)/2) sigma_j C_j.
//
// Row space, block order: per element, the node blocks 0..Nn-1 (each node's terms in
// attachment order), then kin1 (V - (2/T) Q D', nq x Nn, row (j, k) = row0 + k nq + j),
// kin2 (A - (2/T) V D'), then the point rows (off-node / control-point / functional) in
// attachment order; after all elements, the resets.

#include <vector>

#include <Eigen/Dense>

#include "leap/model/contact_set.hpp"

namespace leap {

struct ElementLayout {
  int index = 0;
  NodeDims dims;
  int Nn = 0;     // nodes
  int Nc = 0;     // force control points (0 on an airborne element)
  double T = 0.0; // duration
  double t0 = 0.0;
  bool uStored = true;
  int x0 = 0, nx = 0;                    // primal base offset and size
  int offQ = 0, offV = 0, offA = 0, offLam = -1, offU = -1;  // field offsets relative to x0
  Eigen::MatrixXd E;      // Nn x Nc
  Eigen::VectorXd sigma;  // Nc
  std::vector<int> nodeRow0, nodeRows;
  int kin1Row0 = -1, kin2Row0 = -1;
  int pointRow0 = -1, pointRows = 0;  // the element's point rows (all attachments)

  int xQ(int i, int k) const { return x0 + offQ + k * dims.nq + i; }
  int xV(int i, int k) const { return x0 + offV + k * dims.nv + i; }
  int xA(int i, int k) const { return x0 + offA + k * dims.na + i; }
  int xC(int i, int j) const { return x0 + offLam + j * dims.nlam + i; }
  int xU(int i, int k) const { return x0 + offU + k * dims.nu + i; }
  // Size of one node's decision support [q_k; v_k; a_k; C_0..C_{Nc-1}; u_k].
  int supportSize() const { return dims.nq + dims.nv + dims.na + dims.nlam * Nc + (uStored ? dims.nu : 0); }
  int supportOffC() const { return dims.nq + dims.nv + dims.na; }
  int supportOffU() const { return supportOffC() + dims.nlam * Nc; }
};

struct Layout {
  int np = 0, md = 0;
  std::vector<ElementLayout> elem;
  std::vector<int> resetRow0;
  int n() const { return np + md; }
};

// Maps over one element's fields in y.
struct ElementView {
  Eigen::Map<const Eigen::MatrixXd> Q, V, A, C, U;
  ElementView(const double* y, const ElementLayout& L)
      : Q(y + L.x0 + L.offQ, L.dims.nq, L.Nn),
        V(y + L.x0 + L.offV, L.dims.nv, L.Nn),
        A(y + L.x0 + L.offA, L.dims.na, L.Nn),
        C(L.Nc > 0 ? y + L.x0 + L.offLam : y, L.dims.nlam, L.Nc),
        U(L.uStored ? y + L.x0 + L.offU : y, L.uStored ? L.dims.nu : 0, L.uStored ? L.Nn : 0) {}
};

struct ElementViewMutable {
  Eigen::Map<Eigen::MatrixXd> Q, V, A, C, U;
  ElementViewMutable(double* y, const ElementLayout& L)
      : Q(y + L.x0 + L.offQ, L.dims.nq, L.Nn),
        V(y + L.x0 + L.offV, L.dims.nv, L.Nn),
        A(y + L.x0 + L.offA, L.dims.na, L.Nn),
        C(L.Nc > 0 ? y + L.x0 + L.offLam : y, L.dims.nlam, L.Nc),
        U(L.uStored ? y + L.x0 + L.offU : y, L.uStored ? L.dims.nu : 0, L.uStored ? L.Nn : 0) {}
};

}  // namespace leap
