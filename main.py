import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
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

# Quantum Neural Network
dev = qml.device("default.qubit", wires=n_qubits)

def variational_block(weights):
    for layer in weights:
        for i in range(n_qubits):
            qml.RY(layer[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

@qml.qnode(dev)
def qnn(features, weights):
    qml.AmplitudeEmbedding(features, wires=range(n_qubits), normalize=True)
    variational_block(weights)
    return qml.expval(qml.PauliZ(0))

def binary_cross_entropy(pred, label):
    pred = pnp.clip(pred, 1e-6, 1 - 1e-6)
    return -label * pnp.log(pred) - (1 - label) * pnp.log(1 - pred)

def cost(weights, X, Y):
    loss = 0
    for x, y in zip(X, Y):
        pred = (qnn(x, weights) + 1) / 2
        loss += binary_cross_entropy(pred, y)
    return loss / len(Y)

# Training
depth = 3
weights = pnp.random.normal(0, np.pi, (depth, n_qubits), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.1)

for epoch in range(25):
    weights = opt.step(lambda w: cost(w, X_data, Y_data), weights)
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