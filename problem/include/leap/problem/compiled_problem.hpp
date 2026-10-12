#pragma once

// CompiledProblem: the immutable structure the solver operates on, built once from a
// TrajectoryProblem by compile(): the decision layout, the row space (one term id and a
// sense per row), the resolved per-node term lists, the point rows (off-node /
// control-point / functional) with their interpolation weights, the defect costs and
// the resets. The off-node points of a plain interpolation attachment may be moved in
// place (relocateOffNodeTaus): nothing structural depends on where a point sits.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "leap/problem/layout.hpp"
#include "leap/problem/trajectory_problem.hpp"
#include "leap/spectral/cgl_grid.hpp"

namespace leap {

// Interned term names (one penalty per term).
class TermTable {
public:
  int intern(const std::string& name);
  int size() const { return static_cast<int>(names_.size()); }
  const std::string& name(int id) const { return names_[static_cast<size_t>(id)]; }
  int find(const std::string& name) const;  // -1 if absent
  // The constraint name without the "e<k>/" element prefix.
  static std::string stripped(const std::string& termName);

private:
  std::vector<std::string> names_;
  std::map<std::string, int> index_;
};

struct ResolvedTerm {
  const Constraint* con = nullptr;
  int rowOffset = 0;  // within the node's block
  int rows = 0;
  int term = -1;
};

struct NodeInfo {
  int elem = 0, node = 0;
  std::vector<ResolvedTerm> terms;
  unsigned needs = 0;
  std::vector<int> zIdx;  // decision support [q_k; v_k; a_k; C_0..; u_k]
  int row0 = 0, rows = 0;
};

// One point attachment resolved: nPoints row sets of rowsPerPoint rows each.
struct PointInfo {
  PointKind kind = PointKind::OffNode;
  int elem = 0;
  const Constraint* con = nullptr;
  int term = -1;
  int row0 = 0;
  int rowsPerPoint = 0;
  int nPoints = 0;
  unsigned needs = 0;
  // OffNode: physical state at tau from the element's polynomials.
  std::vector<double> taus;
  std::vector<Eigen::VectorXd> W;     // per point: barycentric weights (Nn) for q, v, a, u
  std::vector<Eigen::VectorXd> Wlam;  // per point: Bernstein * sigma (Nc) for the force
  std::vector<int> zIdxAll;           // [q, v, a, (u)] node-major, then every C column
  int nState = 0;                     // per-node block width in zIdxAll
  // ForceControlPoints / FieldFunctionals: the constraint reads ONLY one field at the
  // linear functional sum_j Wf[p](j) column_j of that field's stored columns.
  Field field = Field::Lam;
  std::vector<Eigen::VectorXd> Wf;            // per point: weights over the field's columns
  std::vector<int> fieldColumns;              // union support (column indices of the field)
  std::vector<std::vector<int>> pointColumns; // per point: positions in fieldColumns with nonzero weight
  std::vector<int> fieldZIdx;                 // decision indices of fieldColumns, field-dim-major
  int fieldDim = 0;
};

struct DefectInfo {
  int elem = 0;
  const Constraint* con = nullptr;
  int rowsPerPoint = 0;
  int nPoints = 0;
  unsigned needs = 0;
  double rho = 0.0;
  std::vector<Eigen::VectorXd> W;     // per point: barycentric (Nn)
  std::vector<Eigen::VectorXd> Wlam;  // per point: Bernstein * sigma (Nc)
  std::vector<double> wq;             // quadrature weights
  std::vector<int> zIdxAll;           // [q, v, a, (u)] node-major, then every C column
  int nState = 0;
};

struct ResetInfo {
  int index = 0;
  const ResetMap* reset = nullptr;
  int tailElem = 0, headElem = 0;
  int row0 = 0, rows = 0;
  int term = -1;
  std::vector<int> zIdx;  // tail node support ++ head node support
};

enum class RowFamily : unsigned char { Node, Kinematic1, Kinematic2, Point, Reset };

struct RowInfo {
  RowFamily family = RowFamily::Node;
  int term = -1;
  int elem = -1;
  int node = -1;      // node rows; -1 otherwise
  double tau = 0.0;   // point rows (off-node); NaN otherwise
  int point = -1;     // point rows: which point of its PointInfo
  int localRow = -1;
  bool inequality = false;
};

class CompiledProblem {
public:
  static CompiledProblem compile(const TrajectoryProblem& spec);

  const RobotModel& model() const { return *model_; }
  std::shared_ptr<const RobotModel> modelPtr() const { return model_; }
  const TrajectoryProblem& spec() const { return spec_; }
  const Layout& layout() const { return layout_; }
  const TermTable& terms() const { return terms_; }
  const std::vector<NodeInfo>& nodes() const { return nodes_; }
  const std::vector<PointInfo>& points() const { return points_; }
  const std::vector<DefectInfo>& defects() const { return defects_; }
  const std::vector<ResetInfo>& resets() const { return resets_; }
  const CglGrid& grid(int elem) const { return *grids_[static_cast<size_t>(elem)]; }
  // d/dt at the nodes of element e: X_dot = a(e) X D'.
  double timeScale(int elem) const { return 2.0 / layout_.elem[static_cast<size_t>(elem)].T; }
  int numElements() const { return static_cast<int>(layout_.elem.size()); }
  int numNodesTotal() const { return static_cast<int>(nodes_.size()); }

  // Row-space arrays (size md).
  const std::vector<int>& rowTerm() const { return rowTerm_; }
  const std::vector<char>& rowIneq() const { return rowIneq_; }
  const std::vector<RowInfo>& rowInfo() const { return rowInfo_; }

  // Move the off-node points of plain interpolation attachment `pointIdx` to `newTaus`
  // (same count); the weights are rebuilt, nothing structural changes. The caller
  // owns the duals of the moved rows (they belong to the old location).
  void relocateOffNodeTaus(int pointIdx, const std::vector<double>& newTaus);

  // Per-node weights on the node COSTS (Clenshaw-Curtis times T/2 turns the node sum
  // into the quadrature of the integral cost). Empty = every weight 1.
  void setNodeCostWeights(std::vector<double> w);
  const std::vector<double>& nodeCostWeights() const { return nodeCostWeight_; }
  void scaleDefectRho(double factor);

  // Physical state of element e at tau from the decision vector (q, v, a, lam, u).
  // u is the stored field, or the inverse-dynamics torque when it is derived (needs a
  // ModelData); lam is the Bezier curve.
  void stateAt(const Eigen::VectorXd& y, int elem, double tau, Eigen::VectorXd& q, Eigen::VectorXd& v,
               Eigen::VectorXd& a, Eigen::VectorXd& lam) const;
  Eigen::VectorXd storedU(const Eigen::VectorXd& y, int elem, double tau) const;  // uStored only
  Eigen::VectorXd physicalControlPoint(const Eigen::VectorXd& y, int elem, int j) const;  // sigma_j C_j
  Eigen::MatrixXd nodeForces(const Eigen::VectorXd& y, int elem) const;  // nlam x Nn

private:
  void buildPointInfo(int e, const PointAttachment& att, PointInfo& info) const;
  void offNodeWeights(int e, double tau, Eigen::VectorXd& W, Eigen::VectorXd& Wlam) const;

  std::shared_ptr<const RobotModel> model_;
  TrajectoryProblem spec_;
  Layout layout_;
  TermTable terms_;
  std::vector<NodeInfo> nodes_;
  std::vector<PointInfo> points_;
  std::vector<DefectInfo> defects_;
  std::vector<ResetInfo> resets_;
  std::vector<std::unique_ptr<CglGrid>> grids_;
  std::vector<int> rowTerm_;
  std::vector<char> rowIneq_;
  std::vector<RowInfo> rowInfo_;
  std::vector<double> nodeCostWeight_;
};

}  // namespace leap
