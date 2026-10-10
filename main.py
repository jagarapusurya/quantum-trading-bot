import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from tradingview_ta import TA_Handler, Interval

# TradingView Setup
symbol = 'RPOWER'
exchange = 'NSE'

handler = TA_Handler(
    symbol=symbol,
    screener="india",
    exchange=exchange,
    interval=Interval.INTERVAL_1_MINUTE
)

# Generate Historical Data (Mock)
prices = np.linspace(2990, 3010, 50)

# Feature Extraction
def extract_features(price):
    volume = 1000000
    volatility = 0.02
    rsi = 55
    return np.array([volume, volatility, rsi, price])

features = np.array([extract_features(price) for price in prices])
features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

# Quantum Setup
n_qubits = 4
n_features = 2 ** n_qubits

X_data, Y_data = [], []

for i in range(len(features_norm) - 1):
    vector = features_norm[i]
    padded = np.pad(vector, (0, n_features - len(vector)), 'constant')
    norm = np.linalg.norm(padded)
    amplitude_vector = padded / norm if norm != 0 else padded

    label = 1 if prices[i + 1] > prices[i] else 0
    X_data.append(amplitude_vector)
    Y_data.append(label)

X_data = np.array(X_data)
Y_data = np.array(Y_data)

# Quantum Neural Network (Qiskit)
depth = 3
def q(wire):
    return n_qubits - 1 - wire

def variational_block(circuit, weights):
    for layer in weights:
        for i in range(n_qubits):
            circuit.ry(layer[i], q(i))
        for i in range(n_qubits - 1):
            circuit.cx(q(i), q(i + 1))
        circuit.cx(q(n_qubits - 1), q(0))

# Parametrised ansatz is built once, then parameters are bound per call
theta = ParameterVector("w", depth * n_qubits)
ansatz = QuantumCircuit(n_qubits)
variational_block(ansatz, np.array(theta, dtype=object).reshape(depth, n_qubits))

observable = SparsePauliOp("".join("Z" if k == q(0) else "I" for k in reversed(range(n_qubits))))

def qnn(features, weights):
    # Amplitude embedding (normalised)
    features = np.asarray(features, dtype=complex)
    features = features / np.linalg.norm(features)
    state = Statevector(features)

    # Variational block
    bound = ansatz.assign_parameters(np.asarray(weights).flatten())
    state = state.evolve(bound)

    # Expectation value of PauliZ on the first wire
    return float(np.real(state.expectation_value(observable)))

def binary_cross_entropy(pred, label):
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return -label * np.log(pred) - (1 - label) * np.log(1 - pred)

def cost(weights, X, Y):
    loss = 0
    for x, y in zip(X, Y):
        pred = (qnn(x, weights) + 1) / 2
        loss += binary_cross_entropy(pred, y)
    return loss / len(Y)

# Gradient via the parameter-shift rule (exact for RY rotations)
def cost_grad(weights, X, Y):
    flat = np.asarray(weights).flatten()
    grad = np.zeros_like(flat)
    shift = np.pi / 2
    for k in range(len(flat)):
        plus, minus = flat.copy(), flat.copy()
        plus[k] += shift
        minus[k] -= shift
        g = 0
        for x, y in zip(X, Y):
            e = qnn(x, flat.reshape(weights.shape))
            p = (e + 1) / 2
            de = (qnn(x, plus.reshape(weights.shape)) - qnn(x, minus.reshape(weights.shape))) / 2
            if 1e-6 < p < 1 - 1e-6:  # clip has zero gradient outside this range
                dloss_dp = -y / p + (1 - y) / (1 - p)
                g += dloss_dp * 0.5 * de
        grad[k] = g / len(Y)
    return grad.reshape(weights.shape)

class AdamOptimizer:
    def __init__(self, stepsize=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.stepsize, self.beta1, self.beta2, self.eps = stepsize, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, grad_fn, weights):
        g = grad_fn(weights)
        self.t += 1
        if self.m is None:
            self.m, self.v = np.zeros_like(weights), np.zeros_like(weights)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g * g
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        return weights - a * self.m / (np.sqrt(self.v) + self.eps)

# Training
weights = np.random.normal(0, np.pi, (depth, n_qubits))
opt = AdamOptimizer(stepsize=0.1)

for epoch in range(25):
    weights = opt.step(lambda w: cost_grad(w, X_data, Y_data), weights)
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {cost(weights, X_data, Y_data):.4f}")

# Live Trading Loop
print("\nStarting live quantum trading bot...")

try:
    while True:
        analysis = handler.get_analysis()

        price = analysis.indicators['close']
        volume = analysis.indicators.get('volume', 1000000)
        rsi = analysis.indicators.get('RSI', 55)

        volatility = 0.02
        features = np.array([volume, volatility, rsi, price])

        features_norm = (features - features.mean()) / (features.std() + 1e-6)
        padded = np.pad(features_norm, (0, n_features - len(features_norm)), 'constant')
        norm = np.linalg.norm(padded)
        amplitude_vector = padded / norm if norm != 0 else padded

        pred = (qnn(amplitude_vector, weights) + 1) / 2
        signal = "BUY" if pred > 0.5 else "SELL"

        print(f"Price: ₹{price:.2f} | Prediction: {pred:.3f} | Signal: {signal}")

        time.sleep(60)

except KeyboardInterrupt:
    print("\nBot stopped by user.")
