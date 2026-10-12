#pragma once

// The solver recipe: every knob of the method, with the defaults of the paper's
// unified recipe (the `unified:` block of a problem yaml, `unified.key=value` on the
// command line). Penalties are expressed in units of the transcription's defect weight
// rho (see applyPenaltyScale), so one set of numbers serves every problem.

#include <string>
#include <vector>

#include "leap/io/yaml_doc.hpp"

namespace leap {

struct SolverRecipe {
  // --- the judge ------------------------------------------------------------------
  double tol = 1e-3;      // criterion on the zero-dual node gate AND the fine grid
  double finedt = 0.01;   // physical-time grid of the fine-grid judge (s)
  bool finegate = true;   // judge the gated fine-grid worst (partial-coverage force rows excluded)

  // --- linear algebra / metric ------------------------------------------------------
  bool gn = true;              // Gauss-Newton metric (no contracted Hessians)
  bool gna0 = true;            // with gn=false: keep Gauss-Newton in the pure-feasibility LM stages (rungs,
                               // A0: zero duals, least squares) and use the exact metric in phase C only;
                               // false = the exact metric everywhere
  bool exactpd = false;        // with gn=false: factorize I + h H by Cholesky and let the step controller
                               // shrink h until it is positive definite (a Levenberg regularization of the
                               // exact Hessian); false = LDLT, the indefinite step is taken and the merit
                               // test decides
  bool sharejunction = true;   // exact junction quotient in every flow
  bool ineqviol = true;        // penalty policies see max(0, g) on inequality rows
  double activemargin = 2e-2;  // anticipatory active set of the metric (row units)
  std::string backend = "block_schur";  // block_schur (alias cholmod_sn) | sparse_cholmod | sparse_llt | sparse_ldlt | dense
  bool cholmod = true;         // CHOLMOD for the sparse factorizations when available
  int threads = 0;             // evaluator threads (0 = OpenMP default)

  // --- objective mode ---------------------------------------------------------------------
  // none: the paper's feasibility mode (phase C on the restoration twin; the posed
  //       objective is reported, not minimized).
  // posed: phase C descends the posed objective f (the problem's cost terms with their
  //        quadrature weights) plus the augmented penalty; the run ends on the level stop
  //        AND, with objstop = stationary, a first-order certificate: the relative reduced
  //        gradient of L_A and the relative complementarity both <= kkttol, and the
  //        objective's relative change <= objtol over objwindows boundaries.
  std::string objective = "none";
  double objtol = 1e-3;
  int objwindows = 2;
  double kkttol = 1e-3;
  std::string objstop = "stationary";  // stationary | feasibility (the criterion alone, f only reshapes phase C)

  // --- step control (every frozen-dual flow) --------------------------------------------
  double h0 = 1e-4, hmin = 2e-5, hshrink = 0.5;
  double hgrow = 2.0;        // ratio controller growth (rho > 0.75 at a full step)
  double ratiolow = 0.25, ratiohigh = 0.75;
  double dymaxrel = 1.0;     // geometric backstop: shrink when ||dz|| > dymaxrel ||z||
  int solveretry = 3;        // shrink-and-retry on a failed factorization
  double blowup = 10.0;      // window blow-up factor over the best window max|c|
  int blowupretry = 3;
  double dsseg = 0.05;       // window length floor (the window is max(dsseg, 4 hmax))
  double window = 0.0;       // > 0: window length in pseudo-time (overrides 4 hmax)
  int stall = 12;            // max|c| plateau patience (windows)
  double stallrel = 1e-2;
  int hcollapse = 4;         // h-collapse abort (windows at hmin without progress)
  double tolflow = 1e-6;     // the flow's own convergence tolerance (gradient certificate)
  double meritstallrel = 1e-3;
  bool nomomentum = false;

  // --- pre-stage: continuation in the obstacle family's geometry --------------------
  std::string pre = "auto";  // none | obsgrow | auto
  std::vector<double> preladder{0.25, 0.5, 0.75};
  double premurel = 0.1;     // rung MuStall seed, in units of rho
  double premumax = 0.0;     // 0 = min(30 premu, mumaxc)
  double pregate = 1e-2;     // rung stop (max|c| of the rung's own problem)
  double presmax = 60.0;
  int presteps = 800;
  int prestall = 8;
  double presmooth = -1.0;   // certificate temperature of the rungs (-1 = posed certificate)

  // --- phase A0: pure-feasibility Levenberg-Marquardt flow (gamma = 0) ---------------
  bool a0 = true;
  double a0murel = 1.0 / 3.0;  // uniform row weight, in units of rho
  double a0smax = 10.0;
  double a0gate = 1e-3;
  int a0steps = 2000;
  double a0window = 1.0;
  int a0stall = 3;
  double a0smooth = 0.003;   // cuboid-certificate temperature during A0 (0 = posed certificate)
  bool a0keepbest = false;
  bool a0momentum = true;

  // --- phase C: classical ALM as frozen-dual LM windows + damped multiplier updates ---
  double mucrel = 0.1;       // per-term MuStall seed, in units of rho
  double mumaxrel = 1.0;     // ... cap
  double mugrow = 3.0;
  double dualstep = 0.5;     // gamma += dualstep mu .* c_eff at every window boundary
  bool dualreset = false;    // reset the step machinery (momentum, look-ahead anchor, h >= h0) after a
                             // multiplier update; false = the history carries across the update (the
                             // look-ahead's merit test is the only guard)
  bool junctionmean = true;  // entry of every flow run: project a discontinuous iterate onto the junction-
                             // group MEAN (the orthogonal projection; the terrain seeds jump in v at
                             // continuity junctions).
                             // false = snap every member to the tail node's value. No effect on a continuous
                             // iterate (fence seeds, prolongations).
  bool mucarry = false;      // carry the per-term penalties across the relocation rounds of a level;
                             // false = every phase C call, round or level, re-seeds every term at muc and
                             // clears the stall history; a level start always re-seeds (Algorithm 1
                             // of the paper)
  double hmax = 5.0;
  double hmaxlevel = 5.0;
  double h0level = 0.05;
  double csmax = 1000.0;     // pseudo-time budget per level
  int csteps = 4000;         // accepted-step cap per level
  int cstepslevel = 0;
  int cwindowsteps = 0;

  // --- level stops ---------------------------------------------------------------------
  double finefrac = 0.5;     // multilevel hand-over: node gate <= finefrac x fine grid
  double finefraccap = 5.0;  // ... and <= finefraccap x tol
  int fineplateau = 6;

  // --- relocation of movable off-node points ------------------------------------------
  bool relocate = true;
  int relocscan = 128;
  double relocminsep = 0.06, reloccover = 0.015, relocmovetol = 1.05;
  double relocfloor = 0.5, relocceil = 60.0, reloctau = 0.95;
  int relocmax = 0;
  bool relocbox = true;
  int rounds = 6;
  double roundsmax = 0.0;
  int roundsteps = 0;

  // --- adaptive local h-refinement -------------------------------------------------------
  int levels = 10;
  double theta = 0.3;
  int maxsplits = 8;
  double splitlo = 0.25, splithi = 0.75;
  bool abortfail = true;
  double abortfactor = 10.0, abortimprove = 2.0;

  bool verbose = true;

  // Resolved absolute penalties (applyPenaltyScale).
  double muc = 3e5, mumaxc = 3e6, a0mu = 1e6, premu = 3e5;
  void applyPenaltyScale(double rho) {
    if (!(rho > 0.0)) return;
    muc = mucrel * rho;
    mumaxc = mumaxrel * rho;
    a0mu = a0murel * rho;
    premu = premurel * rho;
  }
  double windowLength() const { return window > 0.0 ? window : (dsseg > 4.0 * hmax ? dsseg : 4.0 * hmax); }
  std::string fingerprint() const;
};

// The `unified:` block of a problem yaml (and `unified.key=value` overrides) on top of
// the defaults. Some keys of the block exist for compatibility only: they are accepted
// at their default value and rejected otherwise, so a yaml never silently runs a
// different method. Validates the ranges.
SolverRecipe recipeFromYaml(const YamlDoc& d);

}  // namespace leap
