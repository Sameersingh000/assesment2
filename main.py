import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the data (use your actual file path)
data = pd.read_csv('merged_data.csv')
f_t = (data['Signed Volume'] - data['Signed Volume'].mean()) / data['Signed Volume'].std()
sigma = data['mid_price'].std()  # Volatility of mid-price
alpha = 1.67e-4  # Signal strength
phi = 0.139  # Mean-reversion rate of the signal
lambda_ = 0.0035  # Price impact level
beta = 2  # Impact decay rate
T = len(f_t)  # Number of events
gamma_values = np.linspace(0.1, 10, 50)  # Risk aversion parameter range

# Define functions for linear strategy calculation
def compute_coefficients(alpha, phi, sigma, lambda_, beta, gamma):
    sqrt_term = np.sqrt(1 + (2 * lambda_ * beta) / (gamma * sigma**2))
    C_f = (alpha * (1 + beta / phi)) / (gamma * sigma**2 * sqrt_term)
    C_J = sqrt_term - 1
    return C_f, C_J

def simulate_holdings(f_t, C_f, C_J, beta, T):
    J_t = 0  # Initial distortion
    Q_t = np.zeros(T)
    for t in range(T):
        J_t = (1 - beta) * J_t + Q_t[t - 1] if t > 0 else 0
        Q_t[t] = C_f * f_t.iloc[t] - C_J * J_t
    return Q_t

# Function to compute Sharpe Ratio
def compute_sharpe_ratio(strategy, f_t, alpha, sigma):
    pnl = np.mean(strategy * f_t * alpha)  # Profit and Loss
    risk = np.sqrt(np.mean((strategy * sigma) ** 2))  # Risk (Standard deviation)
    return pnl / risk  # Sharpe Ratio

# Build Neural Network Model
def create_nn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare the data for the neural network model
X = f_t.values.reshape(-1, 1)  # Trading signal as input
y = f_t.values  # Using the signal itself as output for simplicity

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the neural network model
nn_model = create_nn_model(X_train.shape[1])
nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Neural network-based strategy prediction
nn_strategy = nn_model.predict(X).flatten()

# Compute Sharpe Ratios for both linear and neural network strategies
sharpe_ratios_linear = []
sharpe_ratios_nn = []

for gamma in gamma_values:
    C_f, C_J = compute_coefficients(alpha, phi, sigma, lambda_, beta, gamma)
    
    # Linear strategy Sharpe Ratio
    Q_t_linear = simulate_holdings(f_t, C_f, C_J, beta, T)
    sharpe_linear = compute_sharpe_ratio(Q_t_linear, f_t, alpha, sigma)
    sharpe_ratios_linear.append(sharpe_linear)

    # Print intermediate values for debugging
    print(f"Gamma: {gamma}, C_f: {C_f}, C_J: {C_J}, Sharpe Linear: {sharpe_linear}")
    
    # Neural network-based strategy Sharpe Ratio
    sharpe_nn = compute_sharpe_ratio(nn_strategy, f_t, alpha, sigma)
    sharpe_ratios_nn.append(sharpe_nn)
    
    # Print intermediate values for debugging
    print(f"Sharpe NN: {sharpe_nn}")

# Plot the comparison of Sharpe Ratios for both strategies
plt.figure(figsize=(10, 6))
plt.plot(gamma_values, sharpe_ratios_linear, label="Linear Strategy Sharpe Ratio", color="blue")
plt.plot(gamma_values, sharpe_ratios_nn, label="Neural Network Strategy Sharpe Ratio", color="green", linestyle='--')
plt.xlabel("Risk Aversion (γ)")
plt.ylabel("Sharpe Ratio")
plt.title("Sharpe Ratio vs Risk Aversion (γ) - Linear vs Neural Network Strategy")
plt.legend()
plt.grid(True)
plt.show()
