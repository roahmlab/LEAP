#include "leap/solver/recipe.hpp"

#include <cstdio>
#include <stdexcept>

#include "leap/linalg/linear_step.hpp"

namespace leap {

namespace {

const std::vector<std::string>& recipeKeys() {
  static const std::vector<std::string> k = {
      // honored
      "tol", "finedt", "finegate", "gn", "gna0", "exactpd", "sharejunction", "ineqviol", "activemargin", "backend", "cholmod", "threads", "h0", "hmin",
      "objective", "objtol", "objwindows", "kkttol", "objstop",
      "hshrink", "hgrow", "ratiolow", "ratiohigh", "dymaxrel", "solveretry", "blowup", "blowupretry", "dsseg", "window",
      "stall", "stallrel", "hcollapse", "tolflow", "meritstallrel", "nomomentum", "pre", "preladder", "premurel", "premumax",
      "pregate", "presmax", "presteps", "prestall", "presmooth", "a0", "a0murel", "a0smax", "a0gate", "a0steps", "a0window",
      "a0stall", "a0smooth", "a0keepbest", "a0momentum", "mucrel", "mumaxrel", "mugrow", "dualstep", "dualreset", "junctionmean", "mucarry", "hmax",
      "hmaxlevel", "h0level", "csmax", "csteps", "cstepslevel", "cwindowsteps", "finefrac", "finefraccap", "fineplateau",
      "relocate", "relocscan", "relocminsep", "reloccover", "relocmovetol", "relocfloor", "relocceil", "reloctau", "relocmax",
      "relocbox", "rounds", "roundsmax", "roundsteps", "levels", "theta", "maxsplits", "splitlo", "splithi", "abortfail",
      "abortfactor", "abortimprove", "verbose",
      // compatibility keys accepted at their default only (see requireFixedDefault)
      "pipeline", "blockmetric", "lowtri", "kron", "pfill", "fastgate", "rowcompact", "lsqmetric", "nla", "nlaoff",
      "eigenthreads", "hyst", "violnear", "nonmonow", "restartmin", "restart", "growgate", "gspike", "probesmax", "premu",
      "a0mu", "murel", "cpure", "muc", "mumaxc", "statgate", "cplateau", "lagmetric", "lagwarm", "laglevel", "dualtransfer",
      "cachedlook", "stopinter", "stopsamp", "internode", "prof", "statrate", "stoprate", "objscale", "objdenom", "objstallrel"};
  return k;
}

// A compatibility key is accepted when it is absent or carries its default value (the
// only value this solver implements).
void requireFixedDefault(const YamlDoc& d, const char* key, const std::string& dflt) {
  const std::string full = std::string("unified.") + key;
  if (!d.has(full)) return;
  const std::string v = d.getS(full, dflt);
  if (v == dflt) return;
  throw std::runtime_error("unified." + std::string(key) + "=" + v + " is a compatibility knob this solver does not carry (only the default '" + dflt + "' is accepted)");
}

void requireFixedDefaultD(const YamlDoc& d, const char* key, double dflt) {
  const std::string full = std::string("unified.") + key;
  if (!d.has(full)) return;
  const double v = d.getD(full, dflt);
  if (v == dflt) return;
  throw std::runtime_error("unified." + std::string(key) + "=" + std::to_string(v) + " is a compatibility knob this solver does not carry (only the default " + std::to_string(dflt) + " is accepted)");
}

void requireFixedDefaultB(const YamlDoc& d, const char* key, bool dflt) {
  const std::string full = std::string("unified.") + key;
  if (!d.has(full)) return;
  if (d.getB(full, dflt) == dflt) return;
  throw std::runtime_error("unified." + std::string(key) + " is a compatibility knob this solver does not carry (only the default " + (dflt ? "true" : "false") + " is accepted)");
}

}  // namespace

SolverRecipe recipeFromYaml(const YamlDoc& d) {
  d.checkKeys("unified", recipeKeys());
  SolverRecipe r;
  const std::string p = "unified.";
  r.tol = d.getD(p + "tol", r.tol);
  r.finedt = d.getD(p + "finedt", r.finedt);
  r.finegate = d.getB(p + "finegate", r.finegate);
  r.gn = d.getB(p + "gn", r.gn);
  r.gna0 = d.getB(p + "gna0", r.gna0);
  r.exactpd = d.getB(p + "exactpd", r.exactpd);
  {
    // the reference exposes 0 (no quotient), 1 (smooth junctions only) and 2 (every
    // junction); 1 and 2 coincide on the problems here (every reset is an identity on q)
    const int sj = d.getI(p + "sharejunction", r.sharejunction ? 2 : 0);
    if (sj < 0 || sj > 2) throw std::runtime_error("unified.sharejunction must be 0, 1 or 2");
    r.sharejunction = sj != 0;
  }
  r.ineqviol = d.getB(p + "ineqviol", r.ineqviol);
  r.activemargin = d.getD(p + "activemargin", r.activemargin);
  r.backend = d.getS(p + "backend", r.backend);
  (void)parseLinearBackend(r.backend);  // validates the name and its availability
  r.cholmod = d.getB(p + "cholmod", r.cholmod);
  r.threads = d.getI(p + "threads", r.threads);
  r.objective = d.getS(p + "objective", r.objective);
  if (r.objective == "effort")
    throw std::runtime_error("unified.objective=effort is not supported: "
                             "pose the objective in the problem yaml's `objective:` block (or attach costs in C++) and run unified.objective=posed");
  if (r.objective != "none" && r.objective != "posed") throw std::runtime_error("unified.objective must be 'none' or 'posed'");
  r.objtol = d.getD(p + "objtol", r.objtol);
  r.objwindows = d.getI(p + "objwindows", r.objwindows);
  r.kkttol = d.getD(p + "kkttol", r.kkttol);
  r.objstop = d.getS(p + "objstop", r.objstop);
  if (r.objstop != "stationary" && r.objstop != "feasibility") throw std::runtime_error("unified.objstop must be 'stationary' or 'feasibility'");
  if (!(r.objtol > 0.0) || r.objwindows < 1 || r.kkttol < 0.0) throw std::runtime_error("unified.objtol > 0, objwindows >= 1, kkttol >= 0");
  r.h0 = d.getD(p + "h0", r.h0);
  r.hmin = d.getD(p + "hmin", r.hmin);
  r.hshrink = d.getD(p + "hshrink", r.hshrink);
  r.hgrow = d.getD(p + "hgrow", r.hgrow);
  r.ratiolow = d.getD(p + "ratiolow", r.ratiolow);
  r.ratiohigh = d.getD(p + "ratiohigh", r.ratiohigh);
  r.dymaxrel = d.getD(p + "dymaxrel", r.dymaxrel);
  r.solveretry = d.getI(p + "solveretry", r.solveretry);
  r.blowup = d.getD(p + "blowup", r.blowup);
  r.blowupretry = d.getI(p + "blowupretry", r.blowupretry);
  r.dsseg = d.getD(p + "dsseg", r.dsseg);
  r.window = d.getD(p + "window", r.window);
  r.stall = d.getI(p + "stall", r.stall);
  r.stallrel = d.getD(p + "stallrel", r.stallrel);
  r.hcollapse = d.getI(p + "hcollapse", r.hcollapse);
  r.tolflow = d.getD(p + "tolflow", r.tolflow);
  r.meritstallrel = d.getD(p + "meritstallrel", r.meritstallrel);
  r.nomomentum = d.getB(p + "nomomentum", r.nomomentum);
  r.pre = d.getS(p + "pre", r.pre);
  if (r.pre != "none" && r.pre != "obsgrow" && r.pre != "auto") throw std::runtime_error("unified.pre must be 'none', 'obsgrow' or 'auto'");
  if (d.has(p + "preladder")) r.preladder = d.getVecD(p + "preladder");
  r.premurel = d.getD(p + "premurel", r.premurel);
  r.premumax = d.getD(p + "premumax", r.premumax);
  r.pregate = d.getD(p + "pregate", r.pregate);
  r.presmax = d.getD(p + "presmax", r.presmax);
  r.presteps = d.getI(p + "presteps", r.presteps);
  r.prestall = d.getI(p + "prestall", r.prestall);
  r.presmooth = d.getD(p + "presmooth", r.presmooth);
  r.a0 = d.getB(p + "a0", r.a0);
  r.a0murel = d.getD(p + "a0murel", r.a0murel);
  r.a0smax = d.getD(p + "a0smax", r.a0smax);
  r.a0gate = d.getD(p + "a0gate", r.a0gate);
  r.a0steps = d.getI(p + "a0steps", r.a0steps);
  r.a0window = d.getD(p + "a0window", r.a0window);
  r.a0stall = d.getI(p + "a0stall", r.a0stall);
  r.a0smooth = d.getD(p + "a0smooth", r.a0smooth);
  r.a0keepbest = d.getB(p + "a0keepbest", r.a0keepbest);
  r.a0momentum = d.getB(p + "a0momentum", r.a0momentum);
  r.mucrel = d.getD(p + "mucrel", r.mucrel);
  r.mumaxrel = d.getD(p + "mumaxrel", r.mumaxrel);
  r.mugrow = d.getD(p + "mugrow", r.mugrow);
  r.dualstep = d.getD(p + "dualstep", r.dualstep);
  r.dualreset = d.getB(p + "dualreset", r.dualreset);
  r.junctionmean = d.getB(p + "junctionmean", r.junctionmean);
  r.mucarry = d.getB(p + "mucarry", r.mucarry);
  r.hmax = d.getD(p + "hmax", r.hmax);
  r.hmaxlevel = d.getD(p + "hmaxlevel", r.hmaxlevel);
  r.h0level = d.getD(p + "h0level", r.h0level);
  r.csmax = d.getD(p + "csmax", r.csmax);
  r.csteps = d.getI(p + "csteps", r.csteps);
  r.cstepslevel = d.getI(p + "cstepslevel", r.cstepslevel);
  r.cwindowsteps = d.getI(p + "cwindowsteps", r.cwindowsteps);
  r.finefrac = d.getD(p + "finefrac", r.finefrac);
  r.finefraccap = d.getD(p + "finefraccap", r.finefraccap);
  r.fineplateau = d.getI(p + "fineplateau", r.fineplateau);
  r.relocate = d.getB(p + "relocate", r.relocate);
  r.relocscan = d.getI(p + "relocscan", r.relocscan);
  r.relocminsep = d.getD(p + "relocminsep", r.relocminsep);
  r.reloccover = d.getD(p + "reloccover", r.reloccover);
  r.relocmovetol = d.getD(p + "relocmovetol", r.relocmovetol);
  r.relocfloor = d.getD(p + "relocfloor", r.relocfloor);
  r.relocceil = d.getD(p + "relocceil", r.relocceil);
  r.reloctau = d.getD(p + "reloctau", r.reloctau);
  r.relocmax = d.getI(p + "relocmax", r.relocmax);
  r.relocbox = d.getB(p + "relocbox", r.relocbox);
  r.rounds = d.getI(p + "rounds", r.rounds);
  r.roundsmax = d.getD(p + "roundsmax", r.roundsmax);
  r.roundsteps = d.getI(p + "roundsteps", r.roundsteps);
  r.levels = d.getI(p + "levels", r.levels);
  r.theta = d.getD(p + "theta", r.theta);
  r.maxsplits = d.getI(p + "maxsplits", r.maxsplits);
  r.splitlo = d.getD(p + "splitlo", r.splitlo);
  r.splithi = d.getD(p + "splithi", r.splithi);
  r.abortfail = d.getB(p + "abortfail", r.abortfail);
  r.abortfactor = d.getD(p + "abortfactor", r.abortfactor);
  r.abortimprove = d.getD(p + "abortimprove", r.abortimprove);
  r.verbose = d.getB(p + "verbose", r.verbose);
  // the compatibility knobs, fixed at their defaults
  requireFixedDefault(d, "pipeline", "unified");
  requireFixedDefaultB(d, "blockmetric", true);
  requireFixedDefaultB(d, "lowtri", true);
  requireFixedDefaultB(d, "kron", true);
  requireFixedDefaultB(d, "pfill", true);
  requireFixedDefaultB(d, "fastgate", true);
  requireFixedDefaultB(d, "rowcompact", true);
  requireFixedDefaultD(d, "lsqmetric", 0);
  requireFixedDefaultB(d, "nla", false);
  requireFixedDefaultD(d, "eigenthreads", 1);
  requireFixedDefaultD(d, "hyst", 0);
  requireFixedDefaultD(d, "violnear", 0);
  requireFixedDefaultD(d, "nonmonow", 0);
  requireFixedDefaultD(d, "restartmin", 0);
  requireFixedDefaultB(d, "restart", false);
  requireFixedDefaultB(d, "growgate", false);
  requireFixedDefaultD(d, "gspike", 0);
  requireFixedDefaultB(d, "murel", true);
  requireFixedDefaultB(d, "cpure", true);
  requireFixedDefaultB(d, "statgate", false);
  requireFixedDefaultB(d, "cplateau", true);
  requireFixedDefaultD(d, "lagmetric", 0);
  requireFixedDefaultD(d, "lagwarm", 0);
  requireFixedDefaultD(d, "laglevel", 0);
  requireFixedDefaultD(d, "dualtransfer", 0);
  requireFixedDefaultB(d, "cachedlook", false);
  requireFixedDefaultD(d, "internode", 0);
  requireFixedDefaultD(d, "statrate", 1e-2);
  requireFixedDefaultD(d, "stoprate", 0);
  requireFixedDefaultD(d, "objdenom", 0);
  requireFixedDefaultD(d, "objstallrel", 1e-3);
  if (d.has(p + "objscale"))
    throw std::runtime_error("unified.objscale is not carried: the objective's scale is the problem's (objective.scale in the yaml)");
  if (d.has(p + "premu") || d.has(p + "a0mu") || d.has(p + "muc") || d.has(p + "mumaxc"))
    throw std::runtime_error("unified.premu/a0mu/muc/mumaxc (absolute penalties) are not supported: use the relative knobs premurel/a0murel/mucrel/mumaxrel");
  // ranges
  if (!(r.tol > 0.0) || !(r.finedt > 0.0)) throw std::runtime_error("unified.tol/finedt must be positive");
  if (!(r.dualstep > 0.0 && r.dualstep <= 4.0)) throw std::runtime_error("unified.dualstep must be in (0, 4]");
  if (!(r.h0 > 0.0) || !(r.hmin > 0.0) || !(r.hmax >= r.hmin)) throw std::runtime_error("unified.h0/hmin/hmax must be positive with hmax >= hmin");
  if (!(r.hshrink > 0.0 && r.hshrink < 1.0) || !(r.hgrow > 1.0)) throw std::runtime_error("unified.hshrink in (0, 1), hgrow > 1");
  if (!(r.ratiolow >= 0.0 && r.ratiolow < r.ratiohigh && r.ratiohigh <= 1.0)) throw std::runtime_error("unified.ratiolow < ratiohigh in [0, 1]");
  if (!(r.theta > 0.0 && r.theta <= 1.0)) throw std::runtime_error("unified.theta must be in (0, 1]");
  if (!(r.splitlo > 0.0 && r.splitlo <= r.splithi && r.splithi < 1.0)) throw std::runtime_error("unified.splitlo <= splithi in (0, 1)");
  if (r.levels < 0 || r.maxsplits < 1 || r.rounds < 0) throw std::runtime_error("unified.levels >= 0, maxsplits >= 1, rounds >= 0");
  if (!(r.mugrow > 1.0) || !(r.mucrel > 0.0) || !(r.mumaxrel >= r.mucrel)) throw std::runtime_error("unified.mugrow > 1, 0 < mucrel <= mumaxrel");
  for (double s : r.preladder)
    if (!(s > 0.0 && s < 1.0)) throw std::runtime_error("unified.preladder entries must lie in (0, 1)");
  return r;
}

std::string SolverRecipe::fingerprint() const {
  char b[3072];
  std::string ladder;
  for (size_t i = 0; i < preladder.size(); ++i) {
    char x[32];
    std::snprintf(x, sizeof(x), "%s%g", i ? "," : "", preladder[i]);
    ladder += x;
  }
  std::snprintf(
      b, sizeof(b),
      "tol=%g finedt=%g finegate=%d gn=%d sharejunction=%d ineqviol=%d activemargin=%g cholmod=%d threads=%d "
      "h0=%g hmin=%g hshrink=%g hgrow=%g ratiolow=%g ratiohigh=%g dymaxrel=%g solveretry=%d blowup=%g blowupretry=%d "
      "dsseg=%g window=%g stall=%d stallrel=%g hcollapse=%d tolflow=%g meritstallrel=%g nomomentum=%d "
      "pre=%s preladder=%s premurel=%g premumax=%g pregate=%g presmax=%g presteps=%d prestall=%d presmooth=%g "
      "a0=%d a0murel=%g a0smax=%g a0gate=%g a0steps=%d a0window=%g a0stall=%d a0smooth=%g a0keepbest=%d a0momentum=%d "
      "mucrel=%g mumaxrel=%g mugrow=%g dualstep=%g dualreset=%d hmax=%g hmaxlevel=%g h0level=%g csmax=%g csteps=%d "
      "cstepslevel=%d cwindowsteps=%d finefrac=%g finefraccap=%g fineplateau=%d relocate=%d relocscan=%d relocminsep=%g "
      "reloccover=%g relocmovetol=%g relocfloor=%g relocceil=%g reloctau=%g relocmax=%d relocbox=%d rounds=%d roundsmax=%g "
      "roundsteps=%d levels=%d theta=%g maxsplits=%d splitlo=%g splithi=%g abortfail=%d abortfactor=%g abortimprove=%g "
      "muc=%g mumaxc=%g a0mu=%g premu=%g",
      tol, finedt, finegate ? 1 : 0, gn ? 1 : 0, sharejunction ? 1 : 0, ineqviol ? 1 : 0, activemargin, cholmod ? 1 : 0, threads,
      h0, hmin, hshrink, hgrow, ratiolow, ratiohigh, dymaxrel, solveretry, blowup, blowupretry, dsseg, window, stall, stallrel,
      hcollapse, tolflow, meritstallrel, nomomentum ? 1 : 0, pre.c_str(), ladder.c_str(), premurel, premumax, pregate, presmax,
      presteps, prestall, presmooth, a0 ? 1 : 0, a0murel, a0smax, a0gate, a0steps, a0window, a0stall, a0smooth, a0keepbest ? 1 : 0,
      a0momentum ? 1 : 0, mucrel, mumaxrel, mugrow, dualstep, dualreset ? 1 : 0, hmax, hmaxlevel, h0level, csmax, csteps,
      cstepslevel, cwindowsteps, finefrac, finefraccap, fineplateau, relocate ? 1 : 0, relocscan, relocminsep, reloccover,
      relocmovetol, relocfloor, relocceil, reloctau, relocmax, relocbox ? 1 : 0, rounds, roundsmax, roundsteps, levels, theta,
      maxsplits, splitlo, splithi, abortfail ? 1 : 0, abortfactor, abortimprove, muc, mumaxc, a0mu, premu);
  std::string out = b;
  // absent at their defaults, so the default recipe's fingerprint is unchanged
  if (backend != "block_schur" && backend != "cholmod_sn") out += " backend=" + backend;
  if (mucarry) out += " mucarry=1";
  if (!junctionmean) out += " junctionmean=0";
  if (!gn) out += std::string(" gna0=") + (gna0 ? "1" : "0") + " exactpd=" + (exactpd ? "1" : "0");
  if (objective != "none") {
    char o[256];
    std::snprintf(o, sizeof(o), " objective=%s objtol=%g objwindows=%d kkttol=%g objstop=%s", objective.c_str(), objtol, objwindows, kkttol,
                  objstop.c_str());
    out += o;
  }
  return out;
}

}  // namespace leap
