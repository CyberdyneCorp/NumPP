// A tiny neural network, end to end, in pure NumPP.
//
// EE/ML concept: a multilayer perceptron (MLP) learns the XOR function, the
// classic example of a problem that a single linear layer CANNOT solve (XOR is
// not linearly separable). A 2->4->1 network with a nonlinear sigmoid hidden
// layer can. We train it with full-batch gradient descent and verify, bit for
// bit, against an identical NumPy training loop.
//
// All weights are initialised to FIXED deterministic values (no RNG) so the run
// is exactly reproducible and the NumPy oracle can replicate it precisely.

#include "parity.hpp"  // ex::check / ex::check_scalar / ex::summary + numpp/numpp.hpp
#include <cstdio>
using namespace numpp;

// Logistic sigmoid s(x) = 1 / (1 + e^-x), built from ufuncs only.
static ndarray sigmoid(const ndarray& x) { return 1.0 / (1.0 + exp(-x)); }

int main() {
  std::printf("== Neural network: 2->4->1 MLP learns XOR (NumPP vs NumPy) ==\n");

  // XOR dataset: 4 samples, 2 inputs, 1 target each.
  ndarray X = zeros({4, 2}, kFloat64);
  X.set_item<double>({0, 0}, 0.0); X.set_item<double>({0, 1}, 0.0);
  X.set_item<double>({1, 0}, 0.0); X.set_item<double>({1, 1}, 1.0);
  X.set_item<double>({2, 0}, 1.0); X.set_item<double>({2, 1}, 0.0);
  X.set_item<double>({3, 0}, 1.0); X.set_item<double>({3, 1}, 1.0);

  ndarray Y = zeros({4, 1}, kFloat64);  // XOR truth table: 0,1,1,0
  Y.set_item<double>({0, 0}, 0.0);
  Y.set_item<double>({1, 0}, 1.0);
  Y.set_item<double>({2, 0}, 1.0);
  Y.set_item<double>({3, 0}, 0.0);

  // ---- Fixed (deterministic) initial weights, chosen to break symmetry ----
  // Layer 1: W1 (2x4), b1 (1x4).  Layer 2: W2 (4x1), b2 (1x1).
  ndarray W1 = zeros({2, 4}, kFloat64);
  double w1_init[2][4] = {{-0.74,  0.32, -0.70, -0.12},
                          { 0.60, -1.00,  0.91,  0.52}};
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j) W1.set_item<double>({i, j}, w1_init[i][j]);

  ndarray b1 = zeros({1, 4}, kFloat64);
  double b1_init[4] = {-0.38, 0.25, 0.54, -0.46};
  for (int j = 0; j < 4; ++j) b1.set_item<double>({0, j}, b1_init[j]);

  ndarray W2 = zeros({4, 1}, kFloat64);
  double w2_init[4] = {-0.01, 0.96, -0.62, 0.87};
  for (int i = 0; i < 4; ++i) W2.set_item<double>({i, 0}, w2_init[i]);

  ndarray b2 = zeros({1, 1}, kFloat64);
  b2.set_item<double>({0, 0}, -0.36);

  // ---- Training: full-batch gradient descent on MSE ----
  const double lr = 0.5;
  const int epochs = 2000;
  ndarray Yhat = zeros({4, 1}, kFloat64);

  for (int e = 0; e < epochs; ++e) {
    // Forward pass.
    ndarray H = sigmoid(matmul(X, W1) + b1);    // hidden activations  (4x4)
    Yhat = sigmoid(matmul(H, W2) + b2);          // output prediction   (4x1)

    // Backward pass. With sigmoid s, s' = s*(1-s).
    ndarray dZ2 = (Yhat - Y) * Yhat * (1.0 - Yhat);          // (4x1)
    ndarray dW2 = matmul(H.transpose(), dZ2);                // (4x1)
    ndarray db2 = sum(dZ2, /*axis=*/0, /*keepdims=*/true);   // (1x1)

    ndarray dH  = matmul(dZ2, W2.transpose());               // (4x4)
    ndarray dZ1 = dH * H * (1.0 - H);                        // (4x4)
    ndarray dW1 = matmul(X.transpose(), dZ1);                // (2x4)
    ndarray db1 = sum(dZ1, /*axis=*/0, /*keepdims=*/true);   // (1x4)

    // Gradient-descent parameter update.
    W2 = W2 - dW2 * lr;  b2 = b2 - db2 * lr;
    W1 = W1 - dW1 * lr;  b1 = b1 - db1 * lr;
  }

  // Final MSE loss as a scalar (mean over all elements -> 0-d array).
  ndarray err = Yhat - Y;
  double mse = mean(err * err).item<double>({});
  std::printf("  final MSE = %.6f\n", mse);

  // ---- (1) Bit-for-bit parity with an identical NumPy training loop ----
  // The PYCODE below replicates the SAME fixed init, lr, epochs and update
  // equations, then sets `a` to the final predictions Yhat.
  const char* py = R"PY(
import numpy as np
def sig(x): return 1.0/(1.0+np.exp(-x))
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = np.array([[0.],[1.],[1.],[0.]])
W1 = np.array([[-0.74,0.32,-0.70,-0.12],
               [0.60,-1.00,0.91,0.52]])
b1 = np.array([[-0.38,0.25,0.54,-0.46]])
W2 = np.array([[-0.01],[0.96],[-0.62],[0.87]])
b2 = np.array([[-0.36]])
lr, epochs = 0.5, 2000
for _ in range(epochs):
    H = sig(X @ W1 + b1)
    Yhat = sig(H @ W2 + b2)
    dZ2 = (Yhat - Y) * Yhat * (1.0 - Yhat)
    dW2 = H.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)
    dH  = dZ2 @ W2.T
    dZ1 = dH * H * (1.0 - H)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)
    W2 = W2 - dW2 * lr; b2 = b2 - db2 * lr
    W1 = W1 - dW1 * lr; b1 = b1 - db1 * lr
a = Yhat
)PY";
  ex::check("Yhat matches NumPy training loop", Yhat, py, 1e-6, 1e-6);

  // ---- (2) Predictions actually solve XOR (within 0.1 of the targets) ----
  ex::check_scalar("pred(0,0) ~ 0", Yhat.item<double>({0, 0}), 0.0, 0.1);
  ex::check_scalar("pred(0,1) ~ 1", Yhat.item<double>({1, 0}), 1.0, 0.1);
  ex::check_scalar("pred(1,0) ~ 1", Yhat.item<double>({2, 0}), 1.0, 0.1);
  ex::check_scalar("pred(1,1) ~ 0", Yhat.item<double>({3, 0}), 0.0, 0.1);

  // ---- Sanity: training reduced the loss to a small value ----
  ex::check_scalar("final MSE below 0.01", mse < 0.01 ? 0.0 : 1.0, 0.0, 0.5);

  return ex::summary();
}
