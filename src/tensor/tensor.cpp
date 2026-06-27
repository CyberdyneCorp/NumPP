#include "numpp/tensor/tensor.hpp"

#include "numpp/backend/backend.hpp"   // matmul
#include "numpp/core/creation.hpp"     // zeros
#include "numpp/core/error.hpp"        // value_error
#include "numpp/core/shape.hpp"        // broadcast_shapes
#include "numpp/linalg/linalg.hpp"     // svdvals

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <vector>

namespace numpp {
namespace {

bool next_index(std::vector<int64_t>& idx, const std::vector<int64_t>& dims) {
  for (int64_t d = static_cast<int64_t>(dims.size()) - 1; d >= 0; --d) {
    if (++idx[d] < dims[d]) return true;
    idx[d] = 0;
  }
  return false;
}

// Expand ellipsis ("...") subscripts into fresh explicit labels, broadcasting
// each operand's ellipsis dimensions to a common shape. Returns the rewritten
// (always explicit, "->") subscript and the broadcast operands; a no-op when no
// "..." appears.
struct Expanded { std::string subs; std::vector<ndarray> ops; };

Expanded expand_ellipsis(const std::string& subscripts, const std::vector<ndarray>& operands) {
  std::string s;
  for (char c : subscripts) if (c != ' ') s += c;
  if (s.find("...") == std::string::npos) return {s, operands};

  const auto arrow = s.find("->");
  const bool explicit_out = arrow != std::string::npos;
  const std::string lhs = explicit_out ? s.substr(0, arrow) : s;
  const std::string rhs = explicit_out ? s.substr(arrow + 2) : std::string();

  std::vector<std::string> terms;
  { std::string cur; for (char c : lhs) { if (c == ',') { terms.push_back(cur); cur.clear(); } else cur += c; } terms.push_back(cur); }

  struct T { std::string pre, post; bool ell; int eni; };
  std::vector<T> ts(terms.size());
  std::array<bool, 128> used{};
  for (size_t i = 0; i < terms.size(); ++i) {
    const std::string& t = terms[i];
    const auto pos = t.find("...");
    T info; info.ell = pos != std::string::npos;
    if (info.ell) { info.pre = t.substr(0, pos); info.post = t.substr(pos + 3); }
    else { info.pre = t; }
    const int explicit_count = static_cast<int>(info.pre.size() + info.post.size());
    info.eni = info.ell ? static_cast<int>(operands[i].ndim()) - explicit_count : 0;
    if (info.eni < 0) throw value_error("einsum: too many subscripts for an operand");
    ts[i] = info;
    for (char c : info.pre) used[static_cast<unsigned char>(c)] = true;
    for (char c : info.post) used[static_cast<unsigned char>(c)] = true;
  }
  for (char c : rhs) if (c != '.') used[static_cast<unsigned char>(c)] = true;

  int E = 0;
  for (const auto& t : ts) E = std::max(E, t.eni);
  Shape ebcast;  // broadcasted ellipsis shape (length E)
  for (size_t i = 0; i < ts.size(); ++i) {
    if (!ts[i].ell || ts[i].eni == 0) continue;
    const Shape& sh = operands[i].shape();
    Shape esub(sh.begin() + ts[i].pre.size(), sh.begin() + ts[i].pre.size() + ts[i].eni);
    ebcast = ebcast.empty() ? esub : broadcast_shapes(ebcast, esub);
  }

  std::string elabels;  // fresh labels for the E ellipsis dims
  { char c = 'A';
    for (int k = 0; k < E; ++k) {
      while (c <= 'z' && (used[static_cast<unsigned char>(c)] || (c > 'Z' && c < 'a'))) ++c;
      if (c > 'z') throw value_error("einsum: ran out of labels for the ellipsis");
      elabels += c; used[static_cast<unsigned char>(c)] = true; ++c;
    } }

  std::vector<ndarray> newops;
  newops.reserve(operands.size());
  std::vector<std::string> newlabels(terms.size());
  for (size_t i = 0; i < ts.size(); ++i) {
    if (ts[i].ell && E > 0) {
      const Shape& sh = operands[i].shape();
      Shape pre_dims(sh.begin(), sh.begin() + ts[i].pre.size());
      Shape post_dims(sh.begin() + ts[i].pre.size() + ts[i].eni, sh.end());
      Shape reshaped = pre_dims;                              // insert (E-eni) ones
      for (int k = 0; k < E - ts[i].eni; ++k) reshaped.push_back(1);
      for (int k = 0; k < ts[i].eni; ++k) reshaped.push_back(sh[ts[i].pre.size() + k]);
      for (int64_t d : post_dims) reshaped.push_back(d);
      Shape target = pre_dims;                                // pre + ebcast + post
      for (int64_t d : ebcast) target.push_back(d);
      for (int64_t d : post_dims) target.push_back(d);
      newops.push_back(operands[i].reshape(reshaped).broadcast_to(target));
      newlabels[i] = ts[i].pre + elabels + ts[i].post;
    } else {
      newops.push_back(operands[i]);
      newlabels[i] = ts[i].pre + ts[i].post;
    }
  }

  std::string outlabels;
  if (explicit_out) {
    const auto pos = rhs.find("...");
    outlabels = pos == std::string::npos ? rhs : rhs.substr(0, pos) + elabels + rhs.substr(pos + 3);
  } else {  // implicit: ellipsis labels first, then free (once-only) labels ascending
    std::array<int, 128> cnt{};
    for (const auto& l : newlabels) for (char c : l) cnt[static_cast<unsigned char>(c)]++;
    std::string freed;
    for (int c = 0; c < 128; ++c)
      if (cnt[c] == 1 && elabels.find(static_cast<char>(c)) == std::string::npos) freed += static_cast<char>(c);
    outlabels = elabels + freed;
  }

  std::string newsubs;
  for (size_t i = 0; i < newlabels.size(); ++i) { if (i) newsubs += ','; newsubs += newlabels[i]; }
  newsubs += "->" + outlabels;
  return {newsubs, newops};
}

struct EinsumPlan {
  std::vector<std::string> in_labels;
  std::string out_labels;
  std::string summed;
  std::array<int64_t, 128> size;
};

EinsumPlan parse_einsum(const std::string& subs, const std::vector<ndarray>& ops) {
  EinsumPlan p;
  p.size.fill(-1);
  std::string s;
  for (char c : subs) if (c != ' ') s += c;  // strip spaces
  const auto arrow = s.find("->");
  const bool explicit_out = arrow != std::string::npos;
  std::string lhs = explicit_out ? s.substr(0, arrow) : s;
  std::string cur;
  for (char c : lhs) {
    if (c == ',') { p.in_labels.push_back(cur); cur.clear(); }
    else cur += c;
  }
  p.in_labels.push_back(cur);
  if (p.in_labels.size() != ops.size()) throw value_error("einsum: operand count mismatch");
  std::array<int, 128> count{};
  for (size_t i = 0; i < ops.size(); ++i) {
    if (static_cast<int64_t>(p.in_labels[i].size()) != ops[i].ndim())
      throw value_error("einsum: label count != operand ndim");
    for (size_t k = 0; k < p.in_labels[i].size(); ++k) {
      const unsigned char ch = static_cast<unsigned char>(p.in_labels[i][k]);
      const int64_t sz = ops[i].shape()[k];
      if (p.size[ch] == -1) p.size[ch] = sz;
      else if (p.size[ch] != sz) throw value_error("einsum: inconsistent dimension for a label");
      ++count[ch];
    }
  }
  if (explicit_out) {
    p.out_labels = s.substr(arrow + 2);
  } else {
    for (int c = 0; c < 128; ++c) if (count[c] == 1) p.out_labels += static_cast<char>(c);  // alphabetical
  }
  std::array<bool, 128> in_out{};
  for (char c : p.out_labels) in_out[static_cast<unsigned char>(c)] = true;
  std::array<bool, 128> seen{};
  for (const auto& lbl : p.in_labels)
    for (char c : lbl) {
      const unsigned char ch = static_cast<unsigned char>(c);
      if (!in_out[ch] && !seen[ch]) { p.summed += c; seen[ch] = true; }
    }
  return p;
}

}  // namespace

ndarray einsum(const std::string& subscripts, const std::vector<ndarray>& operands_in) {
  Expanded ex = expand_ellipsis(subscripts, operands_in);
  const std::vector<ndarray>& operands = ex.ops;
  EinsumPlan p = parse_einsum(ex.subs, operands);
  std::vector<ndarray> ops;
  ops.reserve(operands.size());
  for (const auto& o : operands) ops.push_back(o.astype(kFloat64));

  std::vector<int64_t> out_dims, sum_dims;
  for (char c : p.out_labels) out_dims.push_back(p.size[static_cast<unsigned char>(c)]);
  for (char c : p.summed) sum_dims.push_back(p.size[static_cast<unsigned char>(c)]);

  Shape osh(out_dims.begin(), out_dims.end());
  ndarray out = zeros(osh, kFloat64);

  std::array<int64_t, 128> val{};
  std::vector<int64_t> oi(out_dims.size(), 0);
  do {
    for (size_t d = 0; d < p.out_labels.size(); ++d) val[static_cast<unsigned char>(p.out_labels[d])] = oi[d];
    double acc = 0.0;
    std::vector<int64_t> si(sum_dims.size(), 0);
    do {
      for (size_t d = 0; d < p.summed.size(); ++d) val[static_cast<unsigned char>(p.summed[d])] = si[d];
      double prod = 1.0;
      for (size_t i = 0; i < ops.size(); ++i) {
        std::vector<int64_t> idx;
        idx.reserve(p.in_labels[i].size());
        for (char c : p.in_labels[i]) idx.push_back(val[static_cast<unsigned char>(c)]);
        prod *= ops[i].item<double>(idx);
      }
      acc += prod;
    } while (next_index(si, sum_dims));
    out.set_item<double>(oi, acc);
  } while (next_index(oi, out_dims));
  return out;
}

namespace {

// Labels that must survive when contracting the pair (i,j): anything in the
// final output, or in any other still-present term.
std::string survivors(const std::vector<std::string>& labels, const std::string& out_labels,
                      size_t i, size_t j) {
  std::string keep = out_labels;
  for (size_t k = 0; k < labels.size(); ++k)
    if (k != i && k != j) keep += labels[k];
  return keep;
}

// Ordered, de-duplicated labels of the pair (i,j) that are in `keep`.
std::string pair_result_labels(const std::string& a, const std::string& b, const std::string& keep) {
  std::string c;
  for (char ch : a + b)
    if (keep.find(ch) != std::string::npos && c.find(ch) == std::string::npos) c += ch;
  return c;
}

int64_t labels_size(const std::string& labels, const std::array<int64_t, 128>& size) {
  int64_t n = 1;
  for (char c : labels) n *= size[static_cast<unsigned char>(c)];
  return n;
}

// Greedy pairwise contraction: repeatedly contract the pair whose intermediate
// is smallest, reusing the base einsum for each 1- or 2-operand step. Records the
// chosen (position-i, position-j) pairs into `path` when non-null.
ndarray einsum_greedy(const EinsumPlan& p, std::vector<ndarray> arrs,
                      std::vector<std::vector<int64_t>>* path) {
  std::vector<std::string> labels = p.in_labels;
  const std::string& out = p.out_labels;

  if (arrs.size() == 1)
    return einsum(labels[0] + "->" + out, {arrs[0]});

  while (arrs.size() > 2) {
    size_t bi = 0, bj = 1;
    int64_t best = -1;
    std::string best_c;
    for (size_t i = 0; i < arrs.size(); ++i)
      for (size_t j = i + 1; j < arrs.size(); ++j) {
        const std::string keep = survivors(labels, out, i, j);
        const std::string c = pair_result_labels(labels[i], labels[j], keep);
        const int64_t sz = labels_size(c, p.size);
        if (best == -1 || sz < best) { best = sz; bi = i; bj = j; best_c = c; }
      }
    ndarray inter = einsum(labels[bi] + "," + labels[bj] + "->" + best_c, {arrs[bi], arrs[bj]});
    if (path) path->push_back({static_cast<int64_t>(bi), static_cast<int64_t>(bj)});
    arrs.erase(arrs.begin() + static_cast<std::ptrdiff_t>(bj));
    arrs.erase(arrs.begin() + static_cast<std::ptrdiff_t>(bi));
    labels.erase(labels.begin() + static_cast<std::ptrdiff_t>(bj));
    labels.erase(labels.begin() + static_cast<std::ptrdiff_t>(bi));
    arrs.push_back(inter);
    labels.push_back(best_c);
  }

  if (path) path->push_back({0, 1});
  return einsum(labels[0] + "," + labels[1] + "->" + out, {arrs[0], arrs[1]});
}

}  // namespace

ndarray einsum(const std::string& subscripts, const std::vector<ndarray>& operands_in, bool optimize) {
  if (!optimize) return einsum(subscripts, operands_in);
  Expanded ex = expand_ellipsis(subscripts, operands_in);
  EinsumPlan p = parse_einsum(ex.subs, ex.ops);
  std::vector<ndarray> ops;
  ops.reserve(ex.ops.size());
  for (const auto& o : ex.ops) ops.push_back(o.astype(kFloat64));
  return einsum_greedy(p, ops, nullptr);
}

std::vector<std::vector<int64_t>> einsum_path(const std::string& subscripts,
                                              const std::vector<ndarray>& operands_in) {
  Expanded ex = expand_ellipsis(subscripts, operands_in);
  EinsumPlan p = parse_einsum(ex.subs, ex.ops);
  std::vector<ndarray> ops;
  ops.reserve(ex.ops.size());
  for (const auto& o : ex.ops) ops.push_back(o.astype(kFloat64));
  std::vector<std::vector<int64_t>> path;
  einsum_greedy(p, ops, &path);
  return path;
}

ndarray tensordot(const ndarray& a, const ndarray& b,
                  const std::vector<int64_t>& axes_a, const std::vector<int64_t>& axes_b) {
  std::vector<int64_t> free_a, free_b;
  for (int64_t i = 0; i < a.ndim(); ++i)
    if (std::find(axes_a.begin(), axes_a.end(), i) == axes_a.end()) free_a.push_back(i);
  for (int64_t i = 0; i < b.ndim(); ++i)
    if (std::find(axes_b.begin(), axes_b.end(), i) == axes_b.end()) free_b.push_back(i);

  int64_t N = 1, K = 1, M = 1;
  Shape result_shape;
  std::vector<int64_t> perm_a = free_a, perm_b = axes_b;
  for (int64_t ax : free_a) { N *= a.shape()[ax]; result_shape.push_back(a.shape()[ax]); }
  for (int64_t ax : axes_a) { K *= a.shape()[ax]; perm_a.push_back(ax); }
  for (int64_t ax : free_b) { M *= b.shape()[ax]; result_shape.push_back(b.shape()[ax]); perm_b.push_back(ax); }

  ndarray a2 = a.astype(kFloat64).transpose(perm_a).ascontiguousarray().reshape({N, K});
  ndarray b2 = b.astype(kFloat64).transpose(perm_b).ascontiguousarray().reshape({K, M});
  ndarray r = matmul(a2, b2);
  return result_shape.empty() ? r.reshape({}) : r.reshape(result_shape);
}

ndarray tensordot(const ndarray& a, const ndarray& b, int64_t n) {
  std::vector<int64_t> axes_a, axes_b;
  for (int64_t i = 0; i < n; ++i) { axes_a.push_back(a.ndim() - n + i); axes_b.push_back(i); }
  return tensordot(a, b, axes_a, axes_b);
}

ndarray cross(const ndarray& a, const ndarray& b) {
  ndarray af = a.astype(kFloat64).ascontiguousarray();
  ndarray bf = b.astype(kFloat64).ascontiguousarray();
  const int64_t dim = af.shape().back();
  if (dim != 2 && dim != 3) throw value_error("cross: last axis must have length 2 or 3");
  const int64_t rows = dim ? af.size() / dim : 0;
  const double* pa = af.size() ? af.typed_data<double>() : nullptr;
  const double* pb = bf.size() ? bf.typed_data<double>() : nullptr;
  if (dim == 2) {  // scalar z-component per row
    Shape osh(af.shape().begin(), af.shape().end() - 1);
    ndarray out = osh.empty() ? zeros({}, kFloat64) : zeros(osh, kFloat64);
    double* o = out.size() ? out.typed_data<double>() : nullptr;
    for (int64_t r = 0; r < rows; ++r) o[r] = pa[r * 2] * pb[r * 2 + 1] - pa[r * 2 + 1] * pb[r * 2];
    return out;
  }
  ndarray out = zeros(af.shape(), kFloat64);
  double* o = out.typed_data<double>();
  for (int64_t r = 0; r < rows; ++r) {
    const double* x = pa + r * 3;
    const double* y = pb + r * 3;
    o[r * 3 + 0] = x[1] * y[2] - x[2] * y[1];
    o[r * 3 + 1] = x[2] * y[0] - x[0] * y[2];
    o[r * 3 + 2] = x[0] * y[1] - x[1] * y[0];
  }
  return out;
}

ndarray cond(const ndarray& a) {
  ndarray s = linalg::svdvals(a.astype(kFloat64)).ascontiguousarray();  // descending
  const int64_t n = s.size();
  const double* p = n ? s.typed_data<double>() : nullptr;
  ndarray out = zeros({}, kFloat64);
  out.set_item<double>({}, n ? p[0] / p[n - 1] : 0.0);
  return out;
}

ndarray multi_dot(const std::vector<ndarray>& arrays) {
  if (arrays.empty()) throw value_error("multi_dot: need at least one array");
  ndarray acc = arrays[0];
  for (size_t i = 1; i < arrays.size(); ++i) acc = matmul(acc, arrays[i]);
  return acc;
}

}  // namespace numpp
