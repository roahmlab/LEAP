#include "leap/io/yaml_doc.hpp"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include <fkYAML/node.hpp>

namespace leap {

namespace {

std::vector<std::string> splitOn(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty() || !s.empty()) out.push_back(cur);
  return out;
}

bool allDigits(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s)
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

double strictStrtod(const std::string& s, const std::string& key) {
  const char* b = s.c_str();
  char* e = nullptr;
  const double v = std::strtod(b, &e);
  if (e == b) throw std::runtime_error("yaml: key '" + key + "': not a number: '" + s + "'");
  return v;
}

bool parseBool(const std::string& s, const std::string& key) {
  if (s == "true" || s == "True" || s == "TRUE" || s == "on" || s == "yes") return true;
  if (s == "false" || s == "False" || s == "FALSE" || s == "off" || s == "no") return false;
  return strictStrtod(s, key) != 0.0;
}

double nodeD(const fkyaml::node& n, const std::string& key) {
  if (n.is_float_number()) return n.get_value<double>();
  if (n.is_integer()) return static_cast<double>(n.get_value<std::int64_t>());
  if (n.is_boolean()) return n.get_value<bool>() ? 1.0 : 0.0;
  if (n.is_string()) return strictStrtod(n.get_value<std::string>(), key);
  throw std::runtime_error("yaml: key '" + key + "': not a numeric scalar");
}

bool nodeB(const fkyaml::node& n, const std::string& key) {
  if (n.is_boolean()) return n.get_value<bool>();
  if (n.is_integer()) return n.get_value<std::int64_t>() != 0;
  if (n.is_float_number()) return n.get_value<double>() != 0.0;
  if (n.is_string()) return parseBool(n.get_value<std::string>(), key);
  throw std::runtime_error("yaml: key '" + key + "': not a boolean scalar");
}

std::string nodeS(const fkyaml::node& n, const std::string& key) {
  if (n.is_string()) return n.get_value<std::string>();
  if (n.is_integer()) return std::to_string(n.get_value<std::int64_t>());
  if (n.is_boolean()) return n.get_value<bool>() ? "true" : "false";
  if (n.is_float_number()) {
    std::ostringstream os;
    os << n.get_value<double>();
    return os.str();
  }
  throw std::runtime_error("yaml: key '" + key + "': not a string scalar");
}

}  // namespace

struct YamlDoc::Impl {
  fkyaml::node root;
  std::string text;
  std::string dir;
  std::map<std::string, std::string> ovr;
  std::vector<std::string> ovrList;

  const fkyaml::node* resolve(const std::string& dotted) const {
    const fkyaml::node* n = &root;
    if (dotted.empty()) return n;
    for (const std::string& seg : splitOn(dotted, '.')) {
      if (n->is_mapping()) {
        if (!n->contains(seg)) return nullptr;
        n = &(*n)[seg];
      } else if (n->is_sequence() && allDigits(seg)) {
        const std::size_t idx = static_cast<std::size_t>(std::stoul(seg));
        if (idx >= n->size()) return nullptr;
        n = &(*n)[idx];
      } else {
        return nullptr;
      }
    }
    return n;
  }

  const std::string* override_(const std::string& dotted) const {
    const auto it = ovr.find(dotted);
    return it == ovr.end() ? nullptr : &it->second;
  }
};

YamlDoc::YamlDoc() : impl_(std::make_unique<Impl>()) {}
YamlDoc::YamlDoc(YamlDoc&&) noexcept = default;
YamlDoc& YamlDoc::operator=(YamlDoc&&) noexcept = default;
YamlDoc::~YamlDoc() = default;

YamlDoc YamlDoc::fromFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("yaml: cannot open '" + path + "'");
  std::ostringstream ss;
  ss << in.rdbuf();
  YamlDoc doc = fromString(ss.str());
  doc.impl_->dir = std::filesystem::path(path).parent_path().string();
  return doc;
}

YamlDoc YamlDoc::fromString(const std::string& text) {
  YamlDoc doc;
  doc.impl_->text = text;
  try {
    doc.impl_->root = fkyaml::node::deserialize(text);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("yaml: parse error: ") + e.what());
  }
  if (doc.impl_->root.is_null()) doc.impl_->root = fkyaml::node::mapping();
  if (!doc.impl_->root.is_mapping()) throw std::runtime_error("yaml: document root must be a mapping");
  return doc;
}

bool YamlDoc::has(const std::string& dotted) const {
  return impl_->override_(dotted) != nullptr || impl_->resolve(dotted) != nullptr;
}

double YamlDoc::getD(const std::string& dotted, double dflt) const {
  if (const std::string* o = impl_->override_(dotted)) return strictStrtod(*o, dotted);
  const fkyaml::node* n = impl_->resolve(dotted);
  return n ? nodeD(*n, dotted) : dflt;
}

int YamlDoc::getI(const std::string& dotted, int dflt) const {
  return static_cast<int>(getD(dotted, static_cast<double>(dflt)));
}

bool YamlDoc::getB(const std::string& dotted, bool dflt) const {
  if (const std::string* o = impl_->override_(dotted)) return parseBool(*o, dotted);
  const fkyaml::node* n = impl_->resolve(dotted);
  return n ? nodeB(*n, dotted) : dflt;
}

std::string YamlDoc::getS(const std::string& dotted, const std::string& dflt) const {
  if (const std::string* o = impl_->override_(dotted)) return *o;
  const fkyaml::node* n = impl_->resolve(dotted);
  return n ? nodeS(*n, dotted) : dflt;
}

double YamlDoc::requireD(const std::string& dotted) const {
  if (!has(dotted)) throw std::runtime_error("yaml: required key '" + dotted + "' is missing");
  return getD(dotted, 0.0);
}

int YamlDoc::requireI(const std::string& dotted) const {
  if (!has(dotted)) throw std::runtime_error("yaml: required key '" + dotted + "' is missing");
  return getI(dotted, 0);
}

std::string YamlDoc::requireS(const std::string& dotted) const {
  if (!has(dotted)) throw std::runtime_error("yaml: required key '" + dotted + "' is missing");
  return getS(dotted, "");
}

std::vector<double> YamlDoc::getVecD(const std::string& dotted) const {
  std::vector<double> out;
  if (const std::string* o = impl_->override_(dotted)) {
    for (const std::string& s : splitOn(*o, ',')) out.push_back(strictStrtod(s, dotted));
    return out;
  }
  const fkyaml::node* n = impl_->resolve(dotted);
  if (!n) return out;
  if (!n->is_sequence()) throw std::runtime_error("yaml: key '" + dotted + "': not a sequence");
  for (std::size_t i = 0; i < n->size(); ++i) out.push_back(nodeD((*n)[i], dotted + "." + std::to_string(i)));
  return out;
}

std::vector<std::string> YamlDoc::getVecS(const std::string& dotted) const {
  std::vector<std::string> out;
  if (const std::string* o = impl_->override_(dotted)) return splitOn(*o, ',');
  const fkyaml::node* n = impl_->resolve(dotted);
  if (!n) return out;
  if (!n->is_sequence()) throw std::runtime_error("yaml: key '" + dotted + "': not a sequence");
  for (std::size_t i = 0; i < n->size(); ++i) out.push_back(nodeS((*n)[i], dotted + "." + std::to_string(i)));
  return out;
}

int YamlDoc::seqSize(const std::string& dotted) const {
  const fkyaml::node* n = impl_->resolve(dotted);
  if (!n || !n->is_sequence()) return -1;
  return static_cast<int>(n->size());
}

bool YamlDoc::isMapping(const std::string& dotted) const {
  const fkyaml::node* n = impl_->resolve(dotted);
  return n != nullptr && n->is_mapping();
}

bool YamlDoc::isSequence(const std::string& dotted) const {
  const fkyaml::node* n = impl_->resolve(dotted);
  return n != nullptr && n->is_sequence();
}

std::vector<std::string> YamlDoc::keys(const std::string& dotted) const {
  std::vector<std::string> out;
  const fkyaml::node* n = impl_->resolve(dotted);
  if (!n || !n->is_mapping()) return out;
  for (const auto& item : n->map_items()) out.push_back(item.key().template get_value<std::string>());
  return out;
}

std::string YamlDoc::unknownKeys(const std::string& dotted, const std::vector<std::string>& allowed) const {
  for (const std::string& key : keys(dotted)) {
    bool ok = false;
    for (const std::string& a : allowed) {
      if (key == a) {
        ok = true;
        break;
      }
    }
    if (!ok) return dotted.empty() ? key : dotted + "." + key;
  }
  return "";
}

void YamlDoc::checkKeys(const std::string& dotted, const std::vector<std::string>& allowed) const {
  const std::string bad = unknownKeys(dotted, allowed);
  if (!bad.empty())
    throw std::runtime_error("unknown key '" + bad + "' in section '" + (dotted.empty() ? "(root)" : dotted) + "'");
}

void YamlDoc::setOverride(const std::string& keyEqVal) {
  const std::size_t eq = keyEqVal.find('=');
  if (eq == std::string::npos || eq == 0)
    throw std::runtime_error("yaml: override must be key=value, got '" + keyEqVal + "'");
  impl_->ovr[keyEqVal.substr(0, eq)] = keyEqVal.substr(eq + 1);
  impl_->ovrList.push_back(keyEqVal);
}

const std::vector<std::string>& YamlDoc::overrides() const { return impl_->ovrList; }
const std::string& YamlDoc::sourceText() const { return impl_->text; }
const std::string& YamlDoc::directory() const { return impl_->dir; }

std::string YamlDoc::resolvePath(const std::string& p) const {
  if (p.empty()) return p;
  std::filesystem::path fp(p);
  if (fp.is_absolute() || impl_->dir.empty()) return p;
  return (std::filesystem::path(impl_->dir) / fp).lexically_normal().string();
}

}  // namespace leap
