#pragma once

// Minimal assertion helpers shared by the tests (no framework). A test prints every
// failure, keeps running, and returns the failure count as its exit code.

#include <cmath>
#include <cstdio>

namespace leap::test {

inline int& failures() {
  static int n = 0;
  return n;
}

inline void check(bool cond, const char* file, int line, const char* msg) {
  if (!cond) {
    std::printf("FAIL %s:%d  %s\n", file, line, msg);
    ++failures();
  }
}

inline void checkNear(double a, double b, double tol, const char* file, int line, const char* msg) {
  const double err = std::abs(a - b);
  if (!(err <= tol)) {
    std::printf("FAIL %s:%d  %s  (%.17g vs %.17g, |diff| = %.3e > %.3e)\n", file, line, msg, a, b,
                err, tol);
    ++failures();
  }
}

inline int finish(const char* name) {
  if (failures() == 0)
    std::printf("PASS %s\n", name);
  else
    std::printf("FAILED %s: %d failure(s)\n", name, failures());
  return failures();
}

}  // namespace leap::test

#define CHECK(cond, msg) ::leap::test::check((cond), __FILE__, __LINE__, (msg))
#define CHECK_NEAR(a, b, tol, msg) ::leap::test::checkNear((a), (b), (tol), __FILE__, __LINE__, (msg))
