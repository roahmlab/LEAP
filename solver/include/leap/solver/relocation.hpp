#pragma once

// Relocation of the movable off-node points. Each plain off-node attachment enforces
// its constraint at a few points of its element; between windows the oracle scans the
// constraint's own violation profile on a dense tau grid (values only), finds the
// interior local maxima, and moves the existing points onto the worst of them. Nothing
// structural changes (the layout and the row space do not depend on where a point
// sits); the duals of the moved rows are zeroed.

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/problem/compiled_problem.hpp"

namespace leap {

struct RelocationOptions {
  int nScan = 128;          // dense scan samples per element
  double minSep = 0.06;     // minimum tau separation between chosen targets
  double coverSep = 0.015;  // a target within this of its point is already covered
  double moveTol = 1.05;    // a move must beat the value at the current point by this factor
  double minScaled = 0.0;   // maxima below this (scaled) earn no row
  double maxScaled = 0.0;   // maxima above this (scaled) are never moved onto (0 = none)
  int maxMoves = 0;         // points moved per call (0 = unlimited)
  bool moveBoxRows = true;  // box_u / box_v / motor rows movable too (judged raw)
  double movableTau = 0.95; // |tau| < movableTau
};

struct RelocationReport {
  int scanned = 0, moved = 0;
  double worstScaled = 0.0, worstRaw = 0.0;
  std::string worstName;
  int worstElem = -1;
  double worstTau = 0.0;
};

// `termScales` (size = number of terms, or empty = 1) makes the reported violations
// unit-free. y is the full iterate; the duals of moved rows are zeroed in place.
RelocationReport relocateOffNodePoints(CompiledProblem& prob, Eigen::VectorXd& y, const RelocationOptions& opt,
                                       const Eigen::VectorXd& termScales, bool dryRun);

// Is the fine-grid worst reachable by a relocation round? A movable row set of that
// name must live in that element, the peak must sit inside the movable window and
// below the relocation ceiling.
bool fineGridWorstReachable(const CompiledProblem& prob, const std::string& fineName, int fineElem, double fineTau,
                            double fineValue, double movableTau, double ceiling, bool moveBoxRows);

bool isBoxRowName(const std::string& name);

}  // namespace leap
