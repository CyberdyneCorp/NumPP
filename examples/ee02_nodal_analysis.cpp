#include "parity.hpp"  // ex::check / ex::check_scalar / ex::summary, includes numpp/numpp.hpp
#include <cstdio>
using namespace numpp;

// EE concept: Nodal analysis of a resistive DC circuit.
// Apply Kirchhoff's Current Law (KCL) at each non-ground node. In conductance
// form this yields  G v = I  where:
//   - v is the vector of unknown node voltages,
//   - I is the vector of independent current sources injected into each node,
//   - G is the nodal conductance matrix:
//        G[i][i] = sum of conductances of all resistors touching node i,
//        G[i][j] = -(conductance between node i and node j)   (i != j).
// Because every node has a resistive path to ground, G is symmetric positive
// definite (SPD), so the system has a unique solution.
//
// Resistor network (conductances g = 1/R, in siemens):
//   n1-n2: 1.0   n1-n3: 2.0   n2-n3: 0.5   n2-n4: 1.5
//   n3-n4: 1.0   n1-gnd: 0.5  n4-gnd: 2.0
// Current injections: I = [2, 0, -1, 3] amperes.
int main() {
  std::printf("== EE: Resistive Circuit Nodal Analysis (G v = I) ==\n");

  // --- Build the 4x4 conductance matrix G from the resistor stamps ---
  ndarray G = zeros({4, 4}, kFloat64);
  // Diagonal: total conductance at each node.
  G.set_item<double>({0, 0}, 3.5);  // 1.0 + 2.0 + 0.5(gnd)
  G.set_item<double>({1, 1}, 3.0);  // 1.0 + 0.5 + 1.5
  G.set_item<double>({2, 2}, 3.5);  // 2.0 + 0.5 + 1.0
  G.set_item<double>({3, 3}, 4.5);  // 1.5 + 1.0 + 2.0(gnd)
  // Off-diagonal: negative of the inter-node conductances (symmetric).
  G.set_item<double>({0, 1}, -1.0); G.set_item<double>({1, 0}, -1.0);  // n1-n2
  G.set_item<double>({0, 2}, -2.0); G.set_item<double>({2, 0}, -2.0);  // n1-n3
  G.set_item<double>({1, 2}, -0.5); G.set_item<double>({2, 1}, -0.5);  // n2-n3
  G.set_item<double>({1, 3}, -1.5); G.set_item<double>({3, 1}, -1.5);  // n2-n4
  G.set_item<double>({2, 3}, -1.0); G.set_item<double>({3, 2}, -1.0);  // n3-n4
  // (n1-n4 has no direct resistor, so G[0][3] = G[3][0] = 0.)

  // --- Current source vector I (right-hand side) ---
  ndarray I = zeros({4}, kFloat64);
  I.set_item<double>({0}, 2.0);
  I.set_item<double>({1}, 0.0);
  I.set_item<double>({2}, -1.0);
  I.set_item<double>({3}, 3.0);

  // Shared NumPy definition of the same G and I so the oracle matches exactly.
  const std::string py_setup =
      "G = np.array([[3.5,-1.0,-2.0, 0.0],"
      "              [-1.0, 3.0,-0.5,-1.5],"
      "              [-2.0,-0.5, 3.5,-1.0],"
      "              [ 0.0,-1.5,-1.0, 4.5]]);"
      "I = np.array([2.0, 0.0, -1.0, 3.0]);";

  // --- Solve for node voltages: v = G^{-1} I ---
  ndarray v = linalg::solve(G, I);
  ex::check("node voltages v = solve(G, I)", v,
            py_setup + "a = np.linalg.solve(G, I)");

  // Physical sanity: substitute back. KCL residual G v - I must be ~0,
  // i.e. the currents balance at every node.
  ndarray Gv = dot(G, v);  // matrix-vector product
  ex::check("KCL check: G @ v == I", Gv, py_setup + "a = I");

  // Individual node voltages (scalar parity, with hand-checkable readout).
  ndarray v_np = ex::numpy(py_setup + "a = np.linalg.solve(G, I)");
  ex::check_scalar("v[0] (node 1)", v.item<double>({0}), v_np.item<double>({0}));
  ex::check_scalar("v[3] (node 4)", v.item<double>({3}), v_np.item<double>({3}));

  // --- Determinant of G (nonzero confirms a unique solution / SPD-ish) ---
  ndarray d = linalg::det(G);
  ex::check_scalar("det(G)", d.item<double>({}),
                   ex::numpy(py_setup + "a = np.linalg.det(G)").item<double>({}));

  // --- Inverse conductance matrix = open-circuit resistance matrix ---
  // inv(G)[i][j] is the voltage at node i per unit current injected at node j.
  ndarray Ginv = linalg::inv(G);
  ex::check("inv(G) (resistance matrix)", Ginv,
            py_setup + "a = np.linalg.inv(G)");

  // Consistency: inv(G) @ I must reproduce v as well.
  ndarray v2 = dot(Ginv, I);  // matrix-vector product
  ex::check("inv(G) @ I == v", v2, py_setup + "a = np.linalg.solve(G, I)");

  // inv(G) @ G should be the identity.
  ndarray ident = matmul(Ginv, G);
  ex::check("inv(G) @ G == I4", ident, "a = np.eye(4)");

  // Total power delivered by the sources: P = I . v  (watts).
  double power = dot(I, v).item<double>({});
  ex::check_scalar("delivered power I . v", power,
                   ex::numpy(py_setup + "a = I @ np.linalg.solve(G, I)").item<double>({}));

  return ex::summary();
}
