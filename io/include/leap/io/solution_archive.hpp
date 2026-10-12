#pragma once

// The single-file HDF5 solution archive (docs/solution_format.md): per-element
// trajectories at the nodes (Q, V, A, U, the node forces, the raw force control points
// and their scales), the contact schedule, the reset residuals, the final duals, the
// exact decision vector, the raw residual vector with its row -> term map and the
// per-row penalties, and a JSON metadata blob (configuration echo, model, collision
// geometry, element table, metrics).

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/alm/evaluator.hpp"
#include "leap/problem/compiled_problem.hpp"
#include "leap/solver/flow_driver.hpp"

namespace leap {

struct ArchiveRunInfo {
  std::string generator = "leap_solve";
  std::string problemName, problemType;
  std::string problemYamlText, robotYamlText;
  std::vector<std::string> overrides;
  std::string solverFingerprint, configHashHex;
  std::string urdfPath;
  std::string motorModel = "none";
  int ompThreads = 0;
  double wallS = -1.0;
  double interNodeDynMax = -1.0;  // the fine-grid dynamics worst (the driver's judge)
  const FlowReport* lastFlow = nullptr;
  std::vector<std::string> warnings;
  bool provenance = true;  // the volatile run{timestamp, ...} block
};

// `ev` must be the zero-dual evaluation at y (AlmEvaluator::zeroDualViolation): the
// archive stores ev.g as the raw residual vector and derives the per-node tables and
// the gate from it.
void writeSolutionArchive(const CompiledProblem& prob, const Evaluation& ev, const Eigen::VectorXd& y,
                          const Eigen::VectorXd& muRow, const ArchiveRunInfo& info, const std::string& path);

// FNV-1a of a string as 16 hex digits (the archive's config hash).
std::string fnv1aHex(const std::string& s);

}  // namespace leap
