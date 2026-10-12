#pragma once

// Dotted-path YAML configuration reader (thin wrapper over the vendored fkYAML).
//
//   * keys are addressed by dotted path ("solver.tol"); numeric segments index
//     sequences ("problem.steps.2.tswing");
//   * setOverride("a.b.c=val") registers a command-line override consulted BEFORE the
//     document by every getter; numbers convert through strtod, so an override and
//     the same literal in the file parse to the same double;
//   * missing keys return the caller's default; present-but-unconvertible keys throw
//     std::runtime_error naming the key;
//   * unknownKeys() is the strict-schema check: loaders hard-error on a key outside
//     their whitelist, so a typo never silently falls back to a default.

#include <memory>
#include <string>
#include <vector>

namespace leap {

class YamlDoc {
public:
  static YamlDoc fromFile(const std::string& path);
  static YamlDoc fromString(const std::string& text);

  YamlDoc(YamlDoc&&) noexcept;
  YamlDoc& operator=(YamlDoc&&) noexcept;
  ~YamlDoc();

  bool has(const std::string& dotted) const;
  double getD(const std::string& dotted, double dflt) const;
  int getI(const std::string& dotted, int dflt) const;
  bool getB(const std::string& dotted, bool dflt) const;
  std::string getS(const std::string& dotted, const std::string& dflt) const;
  // Required scalars: throw when the key is absent.
  double requireD(const std::string& dotted) const;
  int requireI(const std::string& dotted) const;
  std::string requireS(const std::string& dotted) const;

  // Sequences of scalars; a missing key gives an empty vector. Overrides use
  // comma-separated values ("problem.step=0.4,0").
  std::vector<double> getVecD(const std::string& dotted) const;
  std::vector<std::string> getVecS(const std::string& dotted) const;
  int seqSize(const std::string& dotted) const;  // -1 when absent or not a sequence
  bool isMapping(const std::string& dotted) const;
  bool isSequence(const std::string& dotted) const;
  // Keys directly under a mapping, in document order.
  std::vector<std::string> keys(const std::string& dotted) const;

  // "" when every key directly under `dotted` ("" = root) is in `allowed`, else the
  // offending key's full dotted path.
  std::string unknownKeys(const std::string& dotted, const std::vector<std::string>& allowed) const;
  // Throws std::runtime_error("unknown key ...") when unknownKeys() is non-empty.
  void checkKeys(const std::string& dotted, const std::vector<std::string>& allowed) const;

  void setOverride(const std::string& keyEqVal);
  const std::vector<std::string>& overrides() const;

  const std::string& sourceText() const;  // verbatim document (archive echo)
  const std::string& directory() const;   // "" for fromString
  // Path value resolved against directory(); absolute paths pass through.
  std::string resolvePath(const std::string& p) const;

private:
  YamlDoc();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace leap
