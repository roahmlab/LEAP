#pragma once

// The exact junction quotient x = E z. The configuration at a junction (the last node
// of one element and the first node of the next) is one unknown for every reset, and
// so is the velocity across a continuity reset. The reset rows that became identities
// stay in the row space with their multipliers pinned to zero; every flow is integrated
// in the reduced coordinates z, with
//     F_z = E' F_x,     H_z = E' H_x E,     M_z = I + h theta H_z,     dx = E dz.
// Disabled (shareJunctions = false) the map is the identity.

#include <vector>

#include <Eigen/Dense>

#include "leap/problem/compiled_problem.hpp"

namespace leap {

class JunctionQuotient {
public:
  JunctionQuotient(const CompiledProblem& prob, bool shareJunctions);

  bool enabled() const { return enabled_; }
  int fullSize() const { return np_; }
  int reducedSize() const { return nz_; }
  // Reduced index of full coordinate i.
  int reducedOf(int i) const { return reducedOf_[static_cast<size_t>(i)]; }
  // Representative (full index) of reduced coordinate r.
  int representative(int r) const { return representative_[static_cast<size_t>(r)]; }
  int multiplicity(int r) const { return multiplicity_[static_cast<size_t>(r)]; }
  // Element that owns reduced coordinate r (the tail element of a shared junction).
  int ownerElement(int r) const { return owner_[static_cast<size_t>(r)]; }

  // F_z = E' F_x (sums the members of a group).
  void reduceCovector(const Eigen::VectorXd& Fx, Eigen::VectorXd& Fz) const;
  // x = E z.
  void scatter(const Eigen::VectorXd& z, Eigen::VectorXd& x) const;
  // Replace every member of a group by the representative's value (snap y onto the
  // quotient manifold) and zero the multipliers of the identity rows.
  void enforce(Eigen::VectorXd& y) const;
  // Replace every member of a group by the group MEAN (the orthogonal projection onto
  // the quotient manifold in the full-coordinate Euclidean norm; the reference tree's
  // `project` at the entry of every flow run) and zero the identity rows' multipliers.
  // Differs from `enforce` only on an off-manifold iterate (a seed whose velocity jumps
  // at a continuity junction, as the terrain seeds do).
  void project(Eigen::VectorXd& y) const;
  // Rows that became identities (continuity resets entirely, plastic-impact q rows).
  const std::vector<int>& identityRows() const { return identityRows_; }
  bool isIdentityRow(int row) const { return identityRow_[static_cast<size_t>(row)] != 0; }
  // True when the whole reset t is an identity under the quotient (continuity).
  bool resetIsIdentity(int t) const { return resetIdentity_[static_cast<size_t>(t)] != 0; }

private:
  bool enabled_;
  int np_ = 0, nz_ = 0, md_ = 0;
  std::vector<int> reducedOf_, representative_, multiplicity_, owner_;
  std::vector<int> identityRows_;
  std::vector<char> identityRow_;
  std::vector<char> resetIdentity_;
};

}  // namespace leap
