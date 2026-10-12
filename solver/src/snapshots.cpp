#include "leap/solver/snapshots.hpp"

#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace leap {

std::vector<double> budgetGrid(double perDecade, double tMin, double tMax) {
  if (!(perDecade > 0.0) || !(tMin > 0.0) || !(tMax >= tMin)) throw std::runtime_error("snapshots: perdecade, tmin and tmax must be positive with tmin <= tmax");
  std::vector<double> grid;
  const double lo = std::floor(std::log10(tMin) * perDecade) / perDecade;
  const double hi = std::log10(tMax);
  for (double e = lo; e <= hi + 1e-9; e += 1.0 / perDecade) {
    const double b = std::pow(10.0, e);
    if (b >= tMin * (1.0 - 1e-12) && b <= tMax * (1.0 + 1e-12)) grid.push_back(b);
  }
  if (grid.empty() || grid.back() < tMax * (1.0 - 1e-9)) grid.push_back(tMax);
  return grid;
}

SnapshotWriter::SnapshotWriter(const SnapshotOptions& options, const ArchiveRunInfo& base)
    : o_(options), base_(base), grid_(budgetGrid(options.perDecade, options.tMin, options.tMax)),
      bestGate_(std::numeric_limits<double>::infinity()) {
  if (o_.dir.empty()) throw std::runtime_error("snapshots: the directory is empty");
  std::filesystem::create_directories(o_.dir);
  index_ = std::fopen((o_.dir + "/snapshots.csv").c_str(), "w");
  stepsCsv_ = std::fopen((o_.dir + "/steps.csv").c_str(), "w");
  if (!index_ || !stepsCsv_) throw std::runtime_error("snapshots: cannot write into " + o_.dir);
  std::fprintf(index_, "index,file,wall_s,step,stage,level,gate,why\n");
  std::fprintf(stepsCsv_, "step,wall_s,stage,level,gate\n");
  base_.lastFlow = nullptr;
  base_.interNodeDynMax = -1.0;
  t0_ = std::chrono::steady_clock::now();
}

SnapshotWriter::~SnapshotWriter() {
  if (index_) std::fclose(index_);
  if (stepsCsv_) std::fclose(stepsCsv_);
}

double SnapshotWriter::wall() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }

double SnapshotWriter::posedGate(Level& L, const Eigen::VectorXd& y) {
  return L.evaluator->zeroDualViolation(y, L.penalties->muRow(), cache_);
}

void SnapshotWriter::write(Level& L, const Eigen::VectorXd& y, double gate, double wallS, long step, int level,
                           const std::string& stage, const char* why) {
  char name[64];
  std::snprintf(name, sizeof(name), "snap_%04ld.h5", written_);
  const std::string path = o_.dir + "/" + name;
  ArchiveRunInfo info = base_;
  info.wallS = wallS;
  Evaluation ev;
  L.evaluator->zeroDualViolation(y, L.penalties->muRow(), ev);
  writeSolutionArchive(*L.restoration, ev, y, L.penalties->muRow(), info, path);
  std::fprintf(index_, "%ld,%s,%.6f,%ld,%s,%d,%.9e,%s\n", written_, name, wallS, step, stage.c_str(), level, gate, why);
  std::fflush(index_);
  ++written_;
}

void SnapshotWriter::begin(Level& L, const Eigen::VectorXd& seed) {
  t0_ = std::chrono::steady_clock::now();
  const double gate = posedGate(L, seed);
  bestY_ = seed;
  bestGate_ = gate;
  bestWall_ = 0.0;
  bestStep_ = 0;
  bestLevel_ = 0;
  bestStage_ = "seed";
  pending_ = !o_.seed;
  if (o_.seed) write(L, seed, gate, 0.0, 0, 0, "seed", "seed");
  std::fprintf(stepsCsv_, "0,0.000000,seed,0,%.9e\n", gate);
  std::printf("[snapshots] %s: %zu budgets from %.4g s to %.6g s (%g per decade); seed gate %.3e\n", o_.dir.c_str(), grid_.size(),
              grid_.front(), grid_.back(), o_.perDecade, gate);
}

void SnapshotWriter::onStep(Level& L, int level, const char* stage, const Eigen::VectorXd& y, double gate) {
  const double t = wall();
  ++steps_;
  std::fprintf(stepsCsv_, "%ld,%.6f,%s,%d,%.9e\n", steps_, t, stage, level, gate);
  // the budgets crossed since the last step: the candidate at each of them is the
  // incumbent of the samples BEFORE this step (all recorded at wall < budget)
  while (nextBudget_ < grid_.size() && t >= grid_[nextBudget_]) {
    if (pending_) {
      write(L, bestY_, bestGate_, bestWall_, bestStep_, bestLevel_, bestStage_, "budget");
      pending_ = false;
    }
    ++nextBudget_;
  }
  if (std::isfinite(gate) && gate < bestGate_ && y.allFinite()) {
    bestY_ = y;
    bestGate_ = gate;
    bestWall_ = t;
    bestStep_ = steps_;
    bestLevel_ = level;
    bestStage_ = stage;
    pending_ = true;
  }
}

void SnapshotWriter::flush(Level& L, int level, const char* why) {
  (void)level;
  if (!pending_) return;
  write(L, bestY_, bestGate_, bestWall_, bestStep_, bestLevel_, bestStage_, why);
  pending_ = false;
}

void SnapshotWriter::finish(Level& L, int level, const Eigen::VectorXd& yFinal, double gateFinal) {
  flush(L, level, "end");
  const double t = wall();
  // the final iterate is always a candidate (the study's end rule keeps the run's last
  // iterate beyond its own wall), even when an earlier iterate had a smaller gate
  if (written_ == 0 || bestStep_ != steps_ || bestLevel_ != level || !(bestGate_ == gateFinal))
    write(L, yFinal, gateFinal, t, steps_, level, "C", "final");
  std::fflush(stepsCsv_);
  std::FILE* f = std::fopen((o_.dir + "/summary.txt").c_str(), "w");
  if (f) {
    std::fprintf(f, "snapshots=%ld\nsteps=%ld\nwall_s=%.6f\nfinal_gate=%.9e\nbudgets=%zu\nperdecade=%g\ntmin=%.17g\ntmax=%.17g\n", written_,
                 steps_, t, gateFinal, grid_.size(), o_.perDecade, o_.tMin, o_.tMax);
    std::fclose(f);
  }
  std::printf("[snapshots] %ld snapshot(s) of %ld accepted steps written to %s (%.1f s)\n", written_, steps_, o_.dir.c_str(), t);
}

}  // namespace leap
