#pragma once

// The staged pipeline (one algorithm, one recipe, every problem):
//
//   level 0 :  cold seed on the coarse mesh
//              [pre-stage]  obstacle-family continuation rungs, iff the seed crosses an obstacle
//              phase A0     pure-feasibility LM flow (gamma = 0, smooth cuboid certificate)
//              phase C      classical ALM: frozen-dual LM windows + damped multiplier updates
//                           + relocation rounds while the fine-grid worst is reachable
//   level k :  judge every element (node rows + gated fine grid); Doerfler marking,
//              bisection at the worst sample, exact prolongation, gamma = 0; phase C + rounds
//   until max(node gate, fine grid) <= tol, a level fails, or the level budget ends.
//
// A ProblemFamily supplies the transcription operators the pipeline cannot know: build
// the current mesh, build an obstacle-modified restoration twin, seed, bisect elements and
// prolong the iterate.

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "leap/alm/evaluator.hpp"
#include "leap/alm/penalties.hpp"
#include "leap/problem/compiled_problem.hpp"
#include "leap/solver/flow_driver.hpp"
#include "leap/solver/judge.hpp"
#include "leap/solver/recipe.hpp"

namespace leap {

// A modification of the obstacle family of the restoration twin (negative = keep).
struct ObstacleVariant {
  double zScale = 1.0;      // every keep-out obstacle scaled vertically about the ground
  double smoothEps = -1.0;  // >= 0: the cuboid certificate's log-sum-exp temperature (0 = exact)
  int rounded = -1;         // 0/1: exact rounded capsule-box distance outside the box
};

struct ElementSplit {
  int element = 0;
  double fraction = 0.5;  // physical-time fraction of the element at which to bisect
};

class ProblemFamily {
public:
  virtual ~ProblemFamily() = default;
  virtual const std::string& type() const = 0;
  virtual const std::string& name() const = 0;
  virtual std::shared_ptr<const RobotModel> model() const = 0;
  // The posed problem on the current mesh (objective + defect cost + every row).
  virtual TrajectoryProblem build() = 0;
  // The current mesh with a modified obstacle family (same layout, same row space).
  virtual TrajectoryProblem obstacleVariant(const ObstacleVariant& v) = 0;
  // The cold seed on the level-0 mesh.
  virtual Eigen::VectorXd seed(const CompiledProblem& prob) = 0;
  // Bisect elements of the current mesh (it becomes the family's current mesh).
  virtual void refine(const std::vector<ElementSplit>& splits) = 0;
  // Exact prolongation of an iterate from the previous mesh onto the current one
  // (primal only; duals are the caller's). Returns the measured prolongation error.
  virtual double prolong(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const CompiledProblem& to,
                         Eigen::VectorXd& yTo) = 0;
  // Unit-free per-term scales for the relocation oracle (empty = 1).
  virtual Eigen::VectorXd oracleTermScales(const CompiledProblem& prob) const;
  // Per-node weights on the node costs that define the posed objective (objective
  // mode). Default: (T_e / 2) x Clenshaw-Curtis, the quadrature of the integral cost.
  virtual std::vector<double> objectiveNodeWeights(const CompiledProblem& prob) const;
  // The transcription's defect weight rho (the unit of every penalty).
  virtual double defectWeight() const = 0;
  virtual int numElements() const = 0;
  virtual const std::vector<std::string>& warnings() const = 0;
  virtual std::string summary() const = 0;
};

// The restoration twin: the same rows, costs removed (the defect cost stays).
TrajectoryProblem restorationTwin(const TrajectoryProblem& posed);

struct Level {
  TrajectoryProblem posedSpec;
  std::unique_ptr<CompiledProblem> posed;        // objective + defect (archive, effort report)
  std::unique_ptr<CompiledProblem> restoration;  // every flow and judge runs here
  std::unique_ptr<AlmEvaluator> evaluator;       // on the restoration twin
  std::unique_ptr<AlmEvaluator> posedEvaluator;
  std::unique_ptr<Penalties> penalties;
  Eigen::VectorXd termScales;
  int elements() const { return restoration ? restoration->numElements() : 0; }
};

struct LevelRecord {
  int level = 0, elements = 0, rounds = 0, moved = 0;
  long cSteps = 0;
  double handoffGate = 0.0, handoffFine = 0.0, gate = 0.0, fine = 0.0, wall = 0.0;
  std::vector<int> marked;
  std::string stop;
};

struct SolveRecord {
  long preSteps = 0, a0Steps = 0;
  double preViol = 0.0, a0Viol = 0.0, preWall = 0.0, a0Wall = 0.0, solveWall = 0.0;
  std::vector<LevelRecord> levels;
  bool met = false;
  std::string stop;
  FlowReport lastFlow;
  // objective mode: the posed objective at the end, its first-order residuals, and
  // whether the final phase C ended on the stationary certificate
  double objective = 0.0, kktRel = 0.0, complementarityRel = 0.0;
  bool objectiveStationary = false;
  long totalSteps() const {
    long n = preSteps + a0Steps;
    for (const LevelRecord& l : levels) n += l.cSteps;
    return n;
  }
};

struct PipelineResult {
  Level level;  // the final level (judged, archived)
  Eigen::VectorXd y;
  SolveRecord record;
};

class SnapshotWriter;  // solver/snapshots.hpp: the budget-study candidates (optional)
PipelineResult runPipeline(ProblemFamily& family, const SolverRecipe& recipe, const std::string& tag,
                           SnapshotWriter* snapshots = nullptr);

// Building blocks (used by the pipeline and by tests).
Level buildLevel(ProblemFamily& family, const SolverRecipe& recipe);
EvaluatorOptions evaluatorOptions(const SolverRecipe& recipe);
FlowConfig phaseCConfig(const SolverRecipe& recipe, int level);
FlowConfig phaseA0Config(const SolverRecipe& recipe);
FlowConfig rungConfig(const SolverRecipe& recipe);

}  // namespace leap
