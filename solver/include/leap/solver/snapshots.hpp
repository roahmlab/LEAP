#pragma once

// Budget-study snapshots: the candidates "a user who stopped at budget b would keep".
//
// The paper's Fig. 3 rolls out, at every compute budget b of a log grid, the iterate with
// the smallest node violation recorded up to b (the "incumbent"). This writer
// watches every accepted step of every stage (obstacle rungs, phase A0, every phase C
// call), keeps the incumbent in memory, and whenever the wall clock crosses a budget of
// the grid it writes the incumbent as an ordinary solution archive (if it changed since
// the last write). The incumbent is also written before the mesh changes (its vector
// lives on the old mesh) and the final iterate is always written last. So the candidate
// at any budget b is: the snapshot with the smallest gate among those with wall <= b,
// which the study's planner reads from `snapshots.csv`.
//
// Files in `dir`: snap_NNNN.h5 (archives, docs/solution_format.md), snapshots.csv
// (index, file, wall_s, step, stage, level, gate, why), steps.csv (every accepted step:
// step, wall_s, stage, level, gate) and summary.txt. The node gate per step is the
// zero-dual max violation of the POSED rows: phase C flows on those rows, so their raw
// per-term violation is the gate; the rungs and a smooth-certificate phase A0 flow on a
// variant problem, so there the posed rows are evaluated once more per step (values
// only).

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/io/solution_archive.hpp"
#include "leap/solver/pipeline.hpp"

namespace leap {

struct SnapshotOptions {
  std::string dir;                         // empty = off
  double perDecade = 24.0;                 // budgets per decade of wall time
  double tMin = 0.042169650342858224;      // the paper's grid: first positive budget (s)
  double tMax = 14398.791664302998;        // ... last budget (s); the grid ends with it
  bool seed = true;                        // write the cold seed as snapshot 0 (budget 0)
};

// The paper's budget grid: 10^(k / perDecade) from the decade step at or below tMin up to
// tMax, tMax appended if it is not on the grid. Budget 0 (the seed) is not part of it.
std::vector<double> budgetGrid(double perDecade, double tMin, double tMax);

class SnapshotWriter {
public:
  SnapshotWriter(const SnapshotOptions& options, const ArchiveRunInfo& base);
  ~SnapshotWriter();
  SnapshotWriter(const SnapshotWriter&) = delete;
  SnapshotWriter& operator=(const SnapshotWriter&) = delete;

  // Starts the wall clock and writes the seed (snapshot 0, budget 0).
  void begin(Level& L, const Eigen::VectorXd& seed);
  // One accepted step of a flow on the mesh of `L`; `gate` is the posed zero-dual node
  // violation of `y`.
  void onStep(Level& L, int level, const char* stage, const Eigen::VectorXd& y, double gate);
  // Writes the incumbent now if it has not been written (call before the mesh changes).
  void flush(Level& L, int level, const char* why);
  // Final flush and the final iterate, then the summary.
  void finish(Level& L, int level, const Eigen::VectorXd& yFinal, double gateFinal);
  // The posed zero-dual gate of y (values only).
  double posedGate(Level& L, const Eigen::VectorXd& y);
  double wall() const;
  long steps() const { return steps_; }
  long written() const { return written_; }
  const std::vector<double>& grid() const { return grid_; }

private:
  void write(Level& L, const Eigen::VectorXd& y, double gate, double wallS, long step, int level, const std::string& stage,
             const char* why);

  SnapshotOptions o_;
  ArchiveRunInfo base_;
  std::chrono::steady_clock::time_point t0_;
  std::vector<double> grid_;
  size_t nextBudget_ = 0;
  long steps_ = 0, written_ = 0;
  // the incumbent (smallest gate so far) and whether it still has to be written
  Eigen::VectorXd bestY_;
  double bestGate_, bestWall_ = 0.0;
  long bestStep_ = 0;
  int bestLevel_ = 0;
  std::string bestStage_;
  bool pending_ = false;
  std::FILE* index_ = nullptr;
  std::FILE* stepsCsv_ = nullptr;
  Evaluation cache_;
};

}  // namespace leap
