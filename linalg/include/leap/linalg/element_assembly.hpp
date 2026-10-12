#pragma once

// ElementMetricAssembly: the part of the linear step every backend shares. The reduced
// coordinates are partitioned into element INTERIORS b = 1..nb and the SEPARATORS s
// (the configuration and velocity of the last node of every element but the last: what
// the neighbouring element and the reset rows reach into). The metric arrives per
// ELEMENT (MetricSink): each element accumulates one dense local matrix over
// [its interior | the separators it touches]; dense blocks are scattered into it,
// Jacobian rows are stacked and turned into one rank-k update at factor time
// (foldRows). Different elements may be assembled concurrently.

#include <vector>

#include <Eigen/Dense>

#include "leap/linalg/linear_step.hpp"

namespace leap {

class ElementMetricAssembly : public LinearStep {
public:
  ElementMetricAssembly(const CompiledProblem& prob, const JunctionQuotient& quotient);
  ~ElementMetricAssembly() override = default;

  int reducedSize() const override { return nz_; }
  int numSeparators() const { return nSep_; }
  void beginAssembly() override;
  void addBlock(int element, const std::vector<int>& idx, const Eigen::MatrixXd& H) override;
  void addWeightedRows(int element, const std::vector<int>& idx, const Eigen::MatrixXd& J, const Eigen::VectorXd& w) override;

protected:
  struct Block {
    std::vector<int> interior;   // reduced coordinates of the interior
    std::vector<int> separators; // global separator ids this element touches
    std::vector<int> sepSlot;    // global separator id -> local slot (or -1)
    int nInt = 0, nLoc = 0;      // local columns: [interior | separators]
    Eigen::MatrixXd Hloc;        // nLoc x nLoc accumulated metric
    Eigen::MatrixXd Jrows;       // stacked weighted-row Jacobians (capacity x nLoc)
    Eigen::VectorXd wrows;
    int nRows = 0;
  };
  struct Entry {  // reduced coordinate -> position
    int element = -1;
    int interiorPos = -1;  // >= 0 when interior
    int separator = -1;    // >= 0 when a separator
  };

  // Hloc += Jrows' diag(w) Jrows (the stacked weighted rows), once per factorization.
  void foldRows(Block& b) const;
  int localColumn(const Block& b, int element, int globalIdx) const;
  void resolveLocal(const Block& b, int element, const std::vector<int>& idx, std::vector<int>& local) const;

  const CompiledProblem* prob_;
  const JunctionQuotient* quotient_;
  int nz_ = 0, nSep_ = 0;
  std::vector<Entry> entry_;
  std::vector<Block> blocks_;
};

}  // namespace leap
