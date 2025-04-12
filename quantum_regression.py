import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import warnings
import argparse

# --- Scikit-learn, XGBoost, LightGBM Imports (with error handling) ---
try:
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.multioutput import MultiOutputRegressor
    sklearn_available = True
except ImportError:
    SVR, KNeighborsRegressor, MultiOutputRegressor = None, None, None
    sklearn_available = False
    print("Warning: scikit-learn not found. SVR and KNeighborsRegressor baselines will be skipped.")

try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    xgb = None
    xgboost_available = False
    print("Warning: xgboost not found. XGBoost baseline will be skipped.")

try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lgb = None
    lightgbm_available = False
    print("Warning: lightgbm not found. LightGBM baseline will be skipped.")
# --- End Imports ---

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Quantum Layer ---
class EnhancedQuantumLayer(nn.Module):
    """
    Improved quantum layer with data re-uploading and trainable rotations
    """
    def __init__(self, n_qubits=8, n_layers=3, input_dim=128): # Reduced defaults
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create a quantum device
        dev = qml.device("default.qubit", wires=n_qubits)
        
        # Define the quantum circuit
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights_rx, weights_ry, weights_rz, weights_cz, final_rotations):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights_rx[l, i], wires=i)
                    qml.RY(weights_ry[l, i], wires=i)
                    qml.RZ(weights_rz[l, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, (i + 1) % n_qubits])
                if l % 2 == 1:
                    for i in range(n_qubits):
                        qml.RY(inputs[i] * weights_cz[l, i], wires=i)
            for i in range(n_qubits):
                qml.RX(final_rotations[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Define weight shapes
        weight_shapes = {
            "weights_rx": (n_layers, n_qubits),
            "weights_ry": (n_layers, n_qubits),
            "weights_rz": (n_layers, n_qubits),
            "weights_cz": (n_layers, n_qubits),
            "final_rotations": n_qubits
        }
        
        # Create the TorchLayer
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Input projection (maps hidden_dim to n_qubits)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, n_qubits),
            nn.LayerNorm(n_qubits)
        )
        
        # Output projection (maps n_qubits back to hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(n_qubits, input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x):
        # Apply input projection
        x_proj = self.input_proj(x)
        
        # Process through the quantum circuit
        results = []
        for sample in x_proj:
            sample_cpu = sample.detach().cpu()
            q_result = self.qlayer(sample_cpu)
            results.append(q_result)
        
        # Stack results and move back to original device
        quantum_output = torch.stack(results).to(x.device)
        
        # Apply output projection
        return self.output_proj(quantum_output)

# --- QNN Regression Model ---
class QuantumRegressionNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_qubits=8, n_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Quantum layer
        self.quantum_layer = EnhancedQuantumLayer(
            n_qubits=n_qubits, n_layers=n_layers, input_dim=hidden_dim
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Quantum processing
        quantum_features = self.quantum_layer(features)
        
        # Combine with skip connection
        combined_features = features + quantum_features
        
        # Output prediction
        output = self.output_layer(combined_features)
        return output

# --- Baseline Models ---
class MLPBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim), # Added LayerNorm
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2), # Added LayerNorm
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Ensure d_model is divisible by nhead
        if hidden_dim % nhead != 0:
            # Find the nearest dimension divisible by nhead
             hidden_dim = (hidden_dim // nhead) * nhead
             if hidden_dim == 0: hidden_dim = nhead # Ensure hidden_dim is at least nhead
             print(f"Adjusting Transformer hidden_dim to {hidden_dim} to be divisible by nhead={nhead}")
             self.input_proj = nn.Linear(input_dim, hidden_dim) # Re-init if changed

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True, dim_feedforward=hidden_dim*2) # Use batch_first=True
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Transformer expects input shape [batch_size, seq_len, features]
        x = self.input_proj(x)
        # Add sequence dimension if it's not present (assuming each sample is a sequence of length 1)
        if x.dim() == 2:
             x = x.unsqueeze(1)
        x = self.transformer(x)
        # Remove sequence dimension before output layer (take the first/only sequence element)
        if x.dim() == 3:
             x = x[:, 0, :] # Select the output for the first token
        return self.output(x)

# --- NEW BASELINES --- 

class CNNBaseline(nn.Module):
    """ Simple 1D CNN for regression on tabular data. """
    def __init__(self, input_dim, output_dim, num_channels=[32, 64], kernel_size=3):
        super().__init__()
        layers = []
        in_channels = 1 # Treat input features as a sequence with 1 channel
        current_dim = input_dim

        for out_channels in num_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_channels))
            # Optional: Add pooling if needed, e.g., nn.MaxPool1d(2) -> would reduce current_dim
            in_channels = out_channels
            # current_dim = current_dim // 2 # Adjust if pooling
        
        # Use Adaptive Avg Pooling to reduce the sequence dimension to 1
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate the flattened size after convolutions and pooling
        # The size is simply the number of output channels from the last conv layer
        # because AdaptiveAvgPool1d(1) reduces the sequence length to 1.
        flattened_size = num_channels[-1]
        
        # Final linear layer
        self.fc = nn.Linear(flattened_size, output_dim)

    def forward(self, x):
        # Input x shape: [batch_size, input_dim]
        # Reshape to [batch_size, channels=1, sequence_length=input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return self.fc(x)

class ResNetBlock(nn.Module):
    """ Basic Residual Block for MLP/ResNet. """
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.norm2(out)
        out += identity # Skip connection
        return self.relu(out)

class ResNetBaseline(nn.Module):
    """ Simple ResNet for regression on tabular data. """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)

class LSTMBaseline(nn.Module):
    """ Simple LSTM for regression on tabular data. """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM expects input shape: (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size=input_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=num_layers, 
                              batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input x shape: [batch_size, input_dim]
        # Reshape to [batch_size, seq_len=1, input_size=input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        # Since seq_len is 1, out[:, -1, :] is equivalent to out[:, 0, :]
        out = self.fc(out[:, -1, :]) 
        return out

# --- QASA Model Components (Adapted from qasa_damped.py) ---

# Define QASA specific quantum device and circuit function
# Note: We might want to make n_qubits/n_layers arguments to the model later
qasa_n_qubits = 8  # Default, can be overridden by args
qasa_n_layers = 4  # Default, can be overridden by args

# Check if Pennylane lightning is available, otherwise use default.qubit
try:
    # Try to initialize lightning.qubit to see if it's available and usable
    qasa_dev = qml.device("lightning.qubit", wires=qasa_n_qubits + 1)
    print("Using Pennylane lightning.qubit for QASA.")
except (qml.DeviceError, ImportError):
    print("Pennylane lightning.qubit not available or failed to initialize. Falling back to default.qubit for QASA.")
    qasa_dev = qml.device("default.qubit", wires=qasa_n_qubits + 1)

@qml.qnode(qasa_dev, interface="torch", diff_method="backprop")
def qasa_quantum_circuit(inputs, weights, n_qubits, n_layers): # Pass n_qubits/n_layers
    # Feature encoding (example)
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)
    
    # Variational layers
    for l in range(n_layers):
        # Layer l operations
        for i in range(n_qubits):
            qml.RX(weights[l, 2*i], wires=i)
            qml.RZ(weights[l, 2*i+1], wires=i)
        
        # Entanglement (example: linear entanglement)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Optional: Entangle last and first qubit for cyclic entanglement
        # qml.CNOT(wires=[n_qubits - 1, 0])
        
        # Optional: Entangle with ancilla (wire n_qubits) - simplified from original
        # If using ancilla, ensure weights shape matches
        # qml.CNOT(wires=[n_qubits - 1, n_qubits]) 
        # qml.RY(weights[l, -1], wires=n_qubits) # Example interaction with ancilla
            
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QASAQuantumLayer(nn.Module):
    """ Adapted Quantum Layer for QASA. """
    def __init__(self, input_dim, output_dim, n_qubits=8, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Calculate weight shape based on the circuit structure
        # Example: 2 rotations per qubit per layer
        num_weights_per_layer = n_qubits * 2 
        # Add weights for ancilla if used (e.g., +1 if RY on ancilla per layer)
        # num_weights_per_layer += 1 
        self.weight_shape = (n_layers, num_weights_per_layer)
        
        # Define the TorchLayer
        self.qlayer = qml.qnn.TorchLayer(qasa_quantum_circuit, {'weights': self.weight_shape})
        
        # Classical projections
        self.input_proj = nn.Linear(input_dim, n_qubits) # Map input to n_qubits
        self.bn = nn.BatchNorm1d(n_qubits) # Normalize quantum input
        self.output_proj = nn.Linear(n_qubits, output_dim) # Map n_qubits back to output_dim

    def forward(self, x):
        # Input x shape: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # Project and normalize input features for the quantum circuit
        x_proj = torch.tanh(self.input_proj(x)) # Tanh activation to constrain inputs
        # BatchNorm expects shape (N, C) or (N, C, L), use (N, C) here
        if batch_size > 1:
             x_proj = self.bn(x_proj)
        elif batch_size == 1: # BatchNorm1d needs N > 1
             # Optionally apply LayerNorm for batch size 1, or skip normalization
             # layernorm = nn.LayerNorm(self.n_qubits).to(x.device)
             # x_proj = layernorm(x_proj)
             pass # Skipping normalization for batch size 1 for simplicity

        # Pass each sample through the quantum layer
        # Important: Ensure qlayer arguments match the qnode definition
        # Also, detach from gpu before passing to Pennylane if needed by backend
        outputs = []
        for i in range(batch_size):
             # Pass n_qubits and n_layers explicitly
             q_out = self.qlayer(x_proj[i].cpu(), n_qubits=self.n_qubits, n_layers=self.n_layers).to(x.device)
             outputs.append(q_out)
        
        # Stack results and project back
        quantum_output = torch.stack(outputs)
        out = self.output_proj(quantum_output)
        
        # Add skip connection
        return x + out # Add quantum residual to original input

class QASAEncoderLayer(nn.Module):
    """ Adapted Quantum Encoder Layer using QASAQuantumLayer. """
    def __init__(self, hidden_dim, nhead=4, dim_feedforward=None, n_qubits=8, n_layers=4):
        super().__init__()
        if dim_feedforward is None:
             dim_feedforward = hidden_dim * 4 # Default feedforward dimension
             
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        # Use the adapted QASAQuantumLayer
        self.quantum_res_block = QASAQuantumLayer(hidden_dim, hidden_dim, n_qubits=n_qubits, n_layers=n_layers)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len=1, hidden_dim]
        
        # Self-Attention part
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Quantum Residual Block + FeedForward part
        # QASAQuantumLayer expects [batch_size, features]
        # Since seq_len is 1, we can squeeze and unsqueeze
        x_squeezed = x.squeeze(1)
        q_out = self.quantum_res_block(x_squeezed) # This includes the skip connection internally
        ffn_out = self.ffn(q_out) 
        # Unsqueeze to add back seq_len dimension before adding to x
        x = self.norm2(x + ffn_out.unsqueeze(1))
        
        return x

class QASARegression(nn.Module):
    """ 
    QASA-style model adapted for tabular regression. 
    Treats each row as a sequence of length 1.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4, nhead=4, 
                 n_qubits=8, q_layers=1): # q_layers: how many of the last layers are quantum
        super().__init__()
        assert q_layers <= num_layers, "Number of quantum layers cannot exceed total layers."
        
        # Input embedding (simple linear layer for tabular data)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        
        # Create encoder layers
        encoder_layers = []
        # Classical layers
        for _ in range(num_layers - q_layers):
            encoder_layers.append(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, 
                                           dim_feedforward=hidden_dim*4, batch_first=True)
            )
        # Quantum layers
        for _ in range(q_layers):
             # Ensure hidden_dim is compatible with nhead for MHA inside QASAEncoderLayer
             if hidden_dim % nhead != 0:
                  compatible_hidden_dim = (hidden_dim // nhead) * nhead
                  if compatible_hidden_dim == 0: compatible_hidden_dim = nhead
                  print(f"Warning: Adjusting QASA hidden_dim {hidden_dim} to {compatible_hidden_dim} for head compatibility.")
                  # If adjustment needed, it implies model definition needs consistent dim
                  # For simplicity, we keep hidden_dim but MHA might error or perform poorly.
                  # A better approach would be to ensure hidden_dim passed is compatible.
                  pass # Keep original hidden_dim for now, MHA might handle it or error
             
             encoder_layers.append(
                 QASAEncoderLayer(hidden_dim=hidden_dim, nhead=nhead, 
                                  n_qubits=n_qubits, n_layers=qasa_n_layers) # Using global qasa_n_layers for quantum circuit depth
             )

        self.encoder = nn.ModuleList(encoder_layers)
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input x shape: [batch_size, input_dim]
        x = self.embedding(x)
        x = self.embedding_norm(x)
        
        # Add sequence dimension: [batch_size, seq_len=1, hidden_dim]
        if x.dim() == 2:
             x = x.unsqueeze(1)
             
        # Pass through encoder layers
        for layer in self.encoder:
            x = layer(x)
            
        # Use the output of the sequence (which is only 1 element long)
        # Shape after encoder: [batch_size, seq_len=1, hidden_dim]
        # Take the element at seq_len=0: [batch_size, hidden_dim]
        if x.dim() == 3:
             x = x[:, 0, :] 
             
        # Final prediction
        output = self.output_layer(x)
        return output

# --- Data Loading (Copied and adapted from quantum_molecular_diffusion.py) ---
def load_molecular_data(excel_path, sheet_name, targets):
    """
    Loads and preprocesses molecular data from an Excel file.
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=1)
        print(f"Successfully loaded Excel file with {len(df)} rows")
        print(f"Columns found: {len(df.columns)}")
        print(f"Data types: {df.dtypes.value_counts().to_dict()}")

        missing_targets = [t for t in targets if t not in df.columns]
        if missing_targets:
            raise ValueError(f"Target columns not found: {missing_targets}")

        print(f"NaN values in target columns before filtering: {df[targets].isna().sum().sum()}")
        df_clean = df.dropna(subset=targets)
        print(f"Remaining rows after dropping NaN targets: {len(df_clean)}")

        input_df = df_clean.select_dtypes(include=[float, int]).drop(columns=targets, errors='ignore')
        feature_columns = input_df.columns.tolist()

        if len(feature_columns) < 5:
            print("Warning: Very few numerical features found.")
        print(f"Input features ({len(feature_columns)}): {feature_columns[:5]}...")

        # Feature engineering (simplified)
        if len(feature_columns) > 1:
             # Simple interaction term (optional, can be expanded)
             try:
                 col1, col2 = feature_columns[0], feature_columns[1]
                 input_df[f'{col1}_x_{col2}'] = df_clean[col1] * df_clean[col2]
                 print("Added one interaction term.")
             except Exception as e:
                 print(f"Warning: Could not create interaction term: {e}")


        if input_df.isna().any().any():
            print("Imputing missing values with median.")
            for col in input_df.columns:
                if input_df[col].isna().any():
                     median_val = input_df[col].median()
                     input_df[col] = input_df[col].fillna(median_val)


        X, Y = input_df.values, df_clean[targets].values

        x_nan_count = np.isnan(X).sum()
        y_nan_count = np.isnan(Y).sum()
        if x_nan_count > 0 or y_nan_count > 0:
            print(f"Warning: Found {x_nan_count} NaN in X, {y_nan_count} in Y after imputation.")

        try:
            X = np.clip(X, np.nanquantile(X, 0.001), np.nanquantile(X, 0.999))
            Y = np.clip(Y, np.nanquantile(Y, 0.001), np.nanquantile(Y, 0.999))
        except Exception as e:
            print(f"Warning: Could not apply quantile clipping: {e}")

        X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        Y = np.nan_to_num(Y, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        print(f"Final data shapes: X: {X.shape}, Y: {Y.shape}")


        stratify = None
        if len(X) > 10:
            try:
                y_mean = np.mean(Y, axis=1)
                n_bins = min(5, len(X) // 5)
                if n_bins > 1:
                    y_strat = pd.qcut(y_mean, n_bins, labels=False, duplicates='drop')
                    if len(set(y_strat)) > 1:
                         stratify = y_strat
            except Exception as e:
                print(f"Warning: Stratification failed: {e}")


        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=stratify
        )

        scaler_x = RobustScaler()
        scaler_y = RobustScaler()

        try:
            X_train_scaled = scaler_x.fit_transform(X_train)
            Y_train_scaled = scaler_y.fit_transform(Y_train)
            X_test_scaled = scaler_x.transform(X_test)
        except Exception as e:
            print(f"Warning: RobustScaler failed: {e}. Falling back to StandardScaler.")
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)
            Y_train_scaled = scaler_y.fit_transform(Y_train)
            X_test_scaled = scaler_x.transform(X_test)
        
        # Keep Y_test unscaled for evaluation, scale Y_train
        Y_test_orig = Y_test.copy()


        for arr, name in [(X_train_scaled, "X_train_scaled"), (Y_train_scaled, "Y_train_scaled"),
                          (X_test_scaled, "X_test_scaled")]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"Warning: Found NaN/Inf in {name} after scaling, replacing with zeros.")
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        # Y_test is kept as numpy array for evaluation function compatibility
        Y_test_numpy = Y_test_orig

        print("Data processing completed successfully.")
        return (X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_numpy,
                scaler_y, input_df.columns.tolist(), targets)

    except Exception as e:
        print(f"Error in data loading: {e}")
        import traceback
        traceback.print_exc()
        raise


# --- Generic Training Function for Baselines and QNN ---
def train_model(
    model, model_name, # Added model_name for saving
    X_train, Y_train, X_val, Y_val,
    epochs=50, batch_size=32, lr=1e-3,
    save_dir_prefix="results", # Use prefix for save dir
    early_stop_patience=5
):
    """
    Generic training function for QNN and Baseline models.
    """
    save_dir = f"{save_dir_prefix}_{model_name}" # Create specific save dir
    os.makedirs(save_dir, exist_ok=True)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    print(f"Starting {model_name} training for {epochs} epochs (saving to {save_dir})...")
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{model_name} Train]", leave=False) as tepoch:
            for i, (x_batch, y_batch) in enumerate(tepoch):
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

                if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                    continue # Skip batch if NaN

                optimizer.zero_grad()
                predictions = model(x_batch)
                
                if torch.isnan(predictions).any():
                     print(f"Warning: NaN prediction in {model_name} epoch {epoch+1}, batch {i}. Skipping.")
                     continue # Skip if prediction is NaN

                loss = loss_fn(predictions, y_batch)
                
                if torch.isnan(loss):
                     print(f"Warning: NaN loss in {model_name} epoch {epoch+1}, batch {i}. Skipping.")
                     continue # Skip if loss is NaN


                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                optimizer.step()
                
                epoch_train_losses.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else float('inf')
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
             with tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [{model_name} Val]", leave=False) as vepoch:
                 for x_val_batch, y_val_batch in vepoch:
                    x_val_batch, y_val_batch = x_val_batch.to(DEVICE), y_val_batch.to(DEVICE)
                    val_preds = model(x_val_batch)
                    val_loss = loss_fn(val_preds, y_val_batch)
                    if not torch.isnan(val_loss):
                         epoch_val_losses.append(val_loss.item())

        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses) if epoch_val_losses else float('inf')
        val_losses.append(avg_val_loss)
        
        scheduler.step() # Step the scheduler

        print(f"Epoch {epoch+1}/{epochs} [{model_name}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'{model_name} Training Loss') # Use model_name in title
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
        plt.close()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_val_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"Epoch {epoch+1} [{model_name}] - Best model saved with validation loss: {best_val_loss:.6f}") # Add print statement
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered for {model_name} after {epoch+1} epochs.")
                if best_model_state: # Check if a best state was ever saved
                    model.load_state_dict(best_model_state) # Load best model
                else:
                    print(f"Warning: Early stopping triggered for {model_name} but no best model state was saved (likely due to NaN losses).")
                break

    # Save final model (even if early stopping occurred)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch, # Save the last epoch number
        'loss': avg_val_loss, # Save the last validation loss
    }, os.path.join(save_dir, 'final_model.pt'))

    print(f"{model_name} training finished.")
    # Load best model state for final return if it exists
    if best_model_state:
         model.load_state_dict(best_model_state)
         print(f"Loaded best model state for {model_name} for final evaluation.")
    return model, train_losses, val_losses


# --- Evaluation Function (Adapted for different model types) ---
def evaluate_predictions(predictions_denorm, Y_test_denorm, target_names, model_name, save_dir): # Added model_name and full save_dir
    """
    Evaluate predictions and create visualizations.
    """
    os.makedirs(save_dir, exist_ok=True) # save_dir is now the full path
    metrics = {target: {} for target in target_names}

    for i, target in enumerate(target_names):
        y_true = Y_test_denorm[:, i]
        y_pred = predictions_denorm[:, i] # Use predictions_denorm directly

        valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        if not np.any(valid_idx):
            print(f"Warning: No valid predictions for {target} in {model_name}")
            metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
            continue

        y_true_valid = y_true[valid_idx]
        y_pred_valid = y_pred[valid_idx]

        if len(y_true_valid) < 2: # Need at least 2 points for R2 score
             print(f"Warning: Not enough valid points for {target} metrics in {model_name}.")
             metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
             continue


        try:
             mse = mean_squared_error(y_true_valid, y_pred_valid)
             rmse = np.sqrt(mse)
             mae = mean_absolute_error(y_true_valid, y_pred_valid)
             r2 = r2_score(y_true_valid, y_pred_valid)

             metrics[target] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

             plt.figure(figsize=(8, 8))
             plt.scatter(y_true_valid, y_pred_valid, alpha=0.5)
             min_val = min(y_true_valid.min(), y_pred_valid.min())
             max_val = max(y_true_valid.max(), y_pred_valid.max())
             plt.plot([min_val, max_val], [min_val, max_val], 'r--')
             plt.xlabel(f"True {target}")
             plt.ylabel(f"Predicted {target}")
             plt.title(f"{target} - {model_name} (R² = {r2:.4f})") # Use model_name in title
             plt.grid(True)
             plt.savefig(os.path.join(save_dir, f"scatter_{target}.png"))
             plt.close()
        except Exception as e:
            print(f"Error calculating/plotting metrics for {target} in {model_name}: {e}")
            metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}


    metrics_df = pd.DataFrame(metrics).T # Transpose for better readability
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"))
    print(f"--- {model_name} Metrics --- ") # Identify model metrics
    print(metrics_df)

    # Save predictions (already denormalized)
    pd.DataFrame(predictions_denorm, columns=target_names).to_csv(
        os.path.join(save_dir, "predictions.csv"), index=False
    )
    print(f"Predictions for {model_name} saved to {os.path.join(save_dir, 'predictions.csv')}")

    return metrics

# --- Result Summarization ---
def summarize_results(all_metrics, target_names, save_dir_prefix="results"):
     """
     Generates comparison plots for metrics across different models.
     all_metrics: A dictionary where keys are model names and values are metric dictionaries.
     """
     print("\n--- Results Summary --- ")
     model_names = list(all_metrics.keys())
     metric_names = ['RMSE', 'MAE', 'R2'] # Focus on these key metrics for summary plot

     summary_data = {metric: {target: [] for target in target_names} for metric in metric_names}
     valid_models_for_metric = {metric: [] for metric in metric_names}

     # Collect data
     for model_name, metrics in all_metrics.items():
         model_has_valid_metrics = False
         for metric in metric_names:
             metric_valid_for_model = True
             for target in target_names:
                  value = metrics.get(target, {}).get(metric, np.nan)
                  if pd.isna(value):
                      metric_valid_for_model = False
                  summary_data[metric][target].append(value) # Append even if NaN for alignment
             if metric_valid_for_model:
                  valid_models_for_metric[metric].append(model_name)
                  model_has_valid_metrics = True
         if not model_has_valid_metrics:
             print(f"Warning: Model '{model_name}' had no valid metrics for summary plots.")


     # Create plots for each metric
     for metric in metric_names:
         plt.figure(figsize=(12, 7))
         num_models = len(model_names)
         bar_width = 0.8 / len(target_names) # Adjust bar width based on number of targets
         index = np.arange(num_models)

         for i, target in enumerate(target_names):
             # Get values, replacing NaN with 0 for plotting, but check validity later
             plot_values = [summary_data[metric][target][j] if not pd.isna(summary_data[metric][target][j]) else 0
                            for j in range(num_models)]
             # Check if any model actually had a valid value for this target/metric combo
             has_valid_data = any(not pd.isna(summary_data[metric][target][j]) for j in range(num_models))

             if has_valid_data:
                  plt.bar(index + i * bar_width, plot_values, bar_width, label=target)
             else:
                  # Still add placeholder for legend if no valid data
                  plt.bar(index + i * bar_width, [0]*num_models, bar_width, label=f'{target} (No Data)')


         plt.xlabel('Model', fontweight='bold')
         plt.ylabel(metric)
         plt.title(f'{metric} Comparison Across Models', fontweight='bold', fontsize=14)
         plt.xticks(index + bar_width * (len(target_names) - 1) / 2, model_names, rotation=15, ha="right")
         plt.legend(title='Targets', bbox_to_anchor=(1.02, 1), loc='upper left')
         plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
         plt.grid(axis='y', linestyle='--', alpha=0.7)
         summary_plot_path = os.path.join(f"{save_dir_prefix}_summary", f"summary_{metric}.png")
         os.makedirs(os.path.dirname(summary_plot_path), exist_ok=True)
         plt.savefig(summary_plot_path)
         print(f"Saved summary plot: {summary_plot_path}")
         plt.close()


# --- Sklearn/XGBoost/LightGBM Pipeline ---
def run_sklearn_pipeline(base_model_class, model_name, model_args, run_flag,
                         X_train_np, Y_train_np, X_test_np, Y_test_np, # Use numpy arrays
                         scaler_y, target_names, save_dir_prefix):
    """
    Runs the training and evaluation pipeline for sklearn-like models.
    Handles multi-output regression using MultiOutputRegressor.
    """
    if not run_flag:
        print(f"\n--- Skipping {model_name} Model --- ")
        return None # Return None if skipped

    # Check if necessary libraries are available
    if model_name in ["SVR", "KNN"] and not sklearn_available:
        print(f"\n--- Skipping {model_name} Model (scikit-learn not available) --- ")
        return None
    if model_name == "XGBoost" and not xgboost_available:
        print(f"\n--- Skipping {model_name} Model (xgboost not available) --- ")
        return None
    if model_name == "LightGBM" and not lightgbm_available:
        print(f"\n--- Skipping {model_name} Model (lightgbm not available) --- ")
        return None
    if not MultiOutputRegressor:
         print(f"\n--- Skipping {model_name} Model (MultiOutputRegressor from sklearn not available) --- ")
         return None

    print(f"\n--- {model_name} Model --- ")
    save_dir = f"{save_dir_prefix}_{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Initialize the base model
        base_model = base_model_class(**model_args)
        # Wrap with MultiOutputRegressor for multi-target prediction
        model = MultiOutputRegressor(base_model)
        print(f"Initialized {model_name} model (wrapped with MultiOutputRegressor):")
        print(model)

        # Train the model
        print(f"Training {model_name} model...")
        # SVR, KNN etc. might expect 1D y for MultiOutputRegressor internally
        # but MultiOutputRegressor handles the looping
        model.fit(X_train_np, Y_train_np) # Use numpy arrays
        print(f"{model_name} training finished.")

        # Generate predictions
        print(f"Generating {model_name} predictions...")
        preds_scaled = model.predict(X_test_np) # Use numpy array

        # Inverse transform predictions
        try:
            preds_denorm = scaler_y.inverse_transform(preds_scaled)
        except Exception as e:
            print(f"Warning: {model_name} inverse transform failed: {e}. Using scaled predictions.")
            preds_denorm = preds_scaled

        # Evaluate predictions
        print(f"Evaluating {model_name} predictions...")
        metrics = evaluate_predictions(
            predictions_denorm=preds_denorm, Y_test_denorm=Y_test_np, # Use numpy Y_test
            target_names=target_names, model_name=model_name, save_dir=save_dir
        )
        return metrics # Return the calculated metrics

    except Exception as e:
        print(f"\n--- Error during {model_name} pipeline: {e} --- ")
        import traceback
        traceback.print_exc()
        return None # Return None on error


# --- Main Execution ---
def main():
    # --- Moved global declaration and device re-init here ---
    global qasa_n_qubits, qasa_n_layers, qasa_dev 
    # --- End Move ---
    
    print("=" * 80)
    print("Quantum & Classical Regression Model Training")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="Quantum and Classical Regression Models")
    # Common args
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (faster training, smaller models)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (for DL models)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (for DL models)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (for DL models)")
    parser.add_argument("--excel-path", type=str, default="pnas.2214357120.sd01.xlsx", help="Path to the Excel data file")
    parser.add_argument("--sheet-name", type=str, default="Dataset", help="Sheet name in the Excel file")
    parser.add_argument("--save-dir-prefix", type=str, default="results", help="Prefix for saving result directories")
    # QNN specific args
    parser.add_argument("--qnn-hidden-dim", type=int, default=256, help="Hidden dimension size for QNN")
    parser.add_argument("--n-qubits", type=int, default=8, help="Number of qubits in Quantum Layer (EnhancedQuantumLayer)")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers in Quantum Layer (EnhancedQuantumLayer)")
    # Baseline specific args (DL)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256, help="Hidden dimension size for MLP")
    parser.add_argument("--transformer-hidden-dim", type=int, default=128, help="Hidden dimension size for Transformer")
    parser.add_argument("--transformer-nhead", type=int, default=4, help="Number of heads for Transformer")
    parser.add_argument("--transformer-nlayers", type=int, default=2, help="Number of layers for Transformer")
    parser.add_argument("--cnn-channels", nargs='+', type=int, default=[32, 64], help="List of channels for CNN layers")
    parser.add_argument("--cnn-kernel-size", type=int, default=3, help="Kernel size for CNN layers")
    parser.add_argument("--resnet-hidden-dim", type=int, default=128, help="Hidden dimension for ResNet")
    parser.add_argument("--resnet-blocks", type=int, default=3, help="Number of residual blocks for ResNet")
    parser.add_argument("--lstm-hidden-dim", type=int, default=128, help="Hidden dimension for LSTM")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of layers for LSTM")
    # Baseline specific args (sklearn/Boosting)
    parser.add_argument("--svr-c", type=float, default=1.0, help="Regularization parameter C for SVR")
    parser.add_argument("--svr-kernel", type=str, default='rbf', help="Kernel type for SVR ('linear', 'rbf', etc.)")
    parser.add_argument("--knn-neighbors", type=int, default=5, help="Number of neighbors for KNN")
    parser.add_argument("--xgb-n-estimators", type=int, default=100, help="Number of boosting rounds for XGBoost")
    parser.add_argument("--xgb-lr", type=float, default=0.1, help="Learning rate for XGBoost")
    parser.add_argument("--lgbm-n-estimators", type=int, default=100, help="Number of boosting rounds for LightGBM")
    parser.add_argument("--lgbm-lr", type=float, default=0.1, help="Learning rate for LightGBM")

    # --- NEW: QASA specific args ---
    # Use the current global values as defaults
    parser.add_argument("--qasa-hidden-dim", type=int, default=128, help="Hidden dimension for QASA Transformer layers")
    parser.add_argument("--qasa-nhead", type=int, default=4, help="Number of heads for QASA attention layers")
    parser.add_argument("--qasa-num-layers", type=int, default=4, help="Total number of encoder layers in QASA")
    parser.add_argument("--qasa-q-layers", type=int, default=1, help="Number of trailing quantum layers in QASA (must be <= qasa-num-layers)")
    parser.add_argument("--qasa-n-qubits", type=int, default=qasa_n_qubits, help=f"Number of qubits for QASA quantum circuit (default: {qasa_n_qubits})")
    parser.add_argument("--qasa-circuit-layers", type=int, default=qasa_n_layers, help=f"Number of layers for QASA quantum circuit (default: {qasa_n_layers})")
    # --- End QASA args ---

    # Flags to run specific models
    parser.add_argument("--run-qnn", action=argparse.BooleanOptionalAction, default=True, help="Run the original QNN model")
    parser.add_argument("--run-mlp", action=argparse.BooleanOptionalAction, default=True, help="Run the MLP baseline model")
    parser.add_argument("--run-transformer", action=argparse.BooleanOptionalAction, default=True, help="Run the Transformer baseline model")
    parser.add_argument("--run-cnn", action=argparse.BooleanOptionalAction, default=True, help="Run the CNN baseline model")
    parser.add_argument("--run-resnet", action=argparse.BooleanOptionalAction, default=True, help="Run the ResNet baseline model")
    parser.add_argument("--run-lstm", action=argparse.BooleanOptionalAction, default=True, help="Run the LSTM baseline model")
    parser.add_argument("--run-qasa", action=argparse.BooleanOptionalAction, default=True, help="Run the QASA model")
    parser.add_argument("--run-svr", action=argparse.BooleanOptionalAction, default=True, help="Run the SVR baseline model")
    parser.add_argument("--run-knn", action=argparse.BooleanOptionalAction, default=True, help="Run the KNN baseline model")
    parser.add_argument("--run-xgboost", action=argparse.BooleanOptionalAction, default=True, help="Run the XGBoost baseline model")
    parser.add_argument("--run-lightgbm", action=argparse.BooleanOptionalAction, default=True, help="Run the LightGBM baseline model")


    args = parser.parse_args()
    
    # --- Update global QASA params AND re-init device based on args --- 
    # global qasa_n_qubits, qasa_n_layers # Already declared global at top
    qasa_n_qubits = args.qasa_n_qubits
    qasa_n_layers = args.qasa_circuit_layers
    # Re-initialize the device with potentially new qubit count
    try:
        qasa_dev = qml.device("lightning.qubit", wires=qasa_n_qubits + 1)
        print(f"QASA device (re)initialized with lightning.qubit for {qasa_n_qubits} qubits (+1 unused ancilla).")
    except (qml.DeviceError, ImportError):
        qasa_dev = qml.device("default.qubit", wires=qasa_n_qubits + 1)
        print(f"QASA device (re)initialized with default.qubit for {qasa_n_qubits} qubits (+1 unused ancilla).")
    # --- End Update --- 

    # Adjust parameters for debug mode
    if args.debug:
        print("\nRunning in DEBUG mode - reduced training parameters")
        epochs = 5 # For DL models
        batch_size = 16 # For DL models
        # QNN
        qnn_hidden_dim = 128
        n_qubits = 4
        n_layers = 2
        # Baselines (DL)
        mlp_hidden_dim = 128
        transformer_hidden_dim = 64
        transformer_nhead= 2
        transformer_nlayers = 1
        cnn_channels = [16, 32]
        cnn_kernel_size = 3
        resnet_hidden_dim = 64
        resnet_blocks = 2
        lstm_hidden_dim = 64
        lstm_layers = 1
        # QASA - Use args values unless debug overrides
        qasa_hidden_dim = 64
        qasa_nhead = 2
        qasa_num_layers = 2
        qasa_q_layers = 1
        # For debug, override QASA qubit/layers regardless of args
        qasa_n_qubits = 4 
        qasa_n_layers = 2 
        print(f"[DEBUG] Overriding QASA circuit params: n_qubits={qasa_n_qubits}, circuit_layers={qasa_n_layers}")
        # Re-init device again for debug override
        try:
            qasa_dev = qml.device("lightning.qubit", wires=qasa_n_qubits + 1)
        except (qml.DeviceError, ImportError):
            qasa_dev = qml.device("default.qubit", wires=qasa_n_qubits + 1)
            
        # Baselines (sklearn/Boosting) - Reduce complexity/estimators
        svr_c = 0.5
        knn_neighbors = 3
        xgb_n_estimators = 20
        lgbm_n_estimators = 20
        patience = 2 # For DL models
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        # QNN
        qnn_hidden_dim = args.qnn_hidden_dim
        n_qubits = args.n_qubits
        n_layers = args.n_layers
        # Baselines (DL)
        mlp_hidden_dim = args.mlp_hidden_dim
        transformer_hidden_dim = args.transformer_hidden_dim
        transformer_nhead = args.transformer_nhead
        transformer_nlayers = args.transformer_nlayers
        cnn_channels = args.cnn_channels
        cnn_kernel_size = args.cnn_kernel_size
        resnet_hidden_dim = args.resnet_hidden_dim
        resnet_blocks = args.resnet_blocks
        lstm_hidden_dim = args.lstm_hidden_dim
        lstm_layers = args.lstm_layers
        # QASA
        qasa_hidden_dim = args.qasa_hidden_dim
        qasa_nhead = args.qasa_nhead
        qasa_num_layers = args.qasa_num_layers
        qasa_q_layers = args.qasa_q_layers
        # No need to assign qasa_n_qubits/qasa_n_layers here, they are global
        # ... (Other non-debug params) ...
        patience = 5 # For DL models

    # Load data
    try:
        target_columns = ['LCE', 'sC', 'aC', 'C']
        print("\nLoading molecular data...")
        (
            X_train, Y_train, X_test, Y_test_numpy,
            scaler_y, feature_names, target_names
        ) = load_molecular_data(args.excel_path, args.sheet_name, target_columns)

        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]

        # Get numpy versions of data for sklearn models
        X_train_np = X_train.numpy()
        # Ensure Y_train is suitable for MultiOutputRegressor (needs 2D)
        Y_train_np = Y_train.numpy() if Y_train.dim() == 2 else Y_train.unsqueeze(1).numpy()
        X_test_np = X_test.numpy()
        # Y_test_numpy is already the correct numpy array

        # Create validation set from training data (Tensors)
        if len(X_train) > 10: # Ensure enough data for validation split
             val_size = min(max(1, int(0.1 * len(X_train))), 500) # 10% validation, max 500 samples
             X_val, Y_val = X_train[-val_size:], Y_train[-val_size:]
             X_train, Y_train = X_train[:-val_size], Y_train[:-val_size]
             print(f"\nUsing {len(X_val)} samples for validation (DL models).")
        else:
             X_val, Y_val = X_train, Y_train # Use all training data for validation if dataset is small
             print("\nWarning: Small dataset, using all training data for validation (DL models).")


        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set (DL): {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Input dimensions: {input_dim}")
        print(f"Output dimensions: {output_dim}")

        all_metrics = {} # Dictionary to store metrics from all models

        # --- Define function to run a DL model pipeline ---
        def run_dl_model_pipeline(model_class, model_name, model_args, run_flag):
            if not run_flag:
                print(f"\n--- Skipping {model_name} Model --- ")
                return

            print(f"\n--- {model_name} Model --- ")
            model = model_class(**model_args)
            print(model)
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{model_name} Model parameters: {num_params:,}")

            trained_model, _, _ = train_model( # Uses the generic DL train function
                model=model, model_name=model_name,
                X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                epochs=epochs, batch_size=batch_size, lr=args.lr,
                save_dir_prefix=args.save_dir_prefix, early_stop_patience=patience
            )

            print(f"Generating {model_name} predictions...")
            trained_model.eval()
            with torch.no_grad():
                preds_scaled = trained_model(X_test.to(DEVICE)).cpu().numpy()
            try:
                preds_denorm = scaler_y.inverse_transform(preds_scaled)
            except Exception as e:
                print(f"Warning: {model_name} inverse transform failed: {e}. Using scaled predictions.")
                preds_denorm = preds_scaled

            print(f"Evaluating {model_name} predictions...")
            save_dir = f"{args.save_dir_prefix}_{model_name}"
            metrics = evaluate_predictions(
                predictions_denorm=preds_denorm, Y_test_denorm=Y_test_numpy,
                target_names=target_names, model_name=model_name, save_dir=save_dir
            )
            if metrics: # Store metrics if evaluation was successful
                all_metrics[model_name] = metrics

        # --- Run DL Model Pipelines ---
        run_dl_model_pipeline(
            QuantumRegressionNet, "QNN",
            {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': qnn_hidden_dim, 'n_qubits': n_qubits, 'n_layers': n_layers},
            args.run_qnn
        )
        run_dl_model_pipeline(
            MLPBaseline, "MLP",
            {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': mlp_hidden_dim},
            args.run_mlp
        )
        # Adjust transformer hidden_dim if needed before creating args
        current_transformer_hidden_dim = transformer_hidden_dim
        if args.run_transformer and current_transformer_hidden_dim % transformer_nhead != 0:
            original_dim = current_transformer_hidden_dim
            current_transformer_hidden_dim = (current_transformer_hidden_dim // transformer_nhead) * transformer_nhead
            if current_transformer_hidden_dim == 0: current_transformer_hidden_dim = transformer_nhead
            print(f"Adjusted Transformer hidden_dim from {original_dim} to {current_transformer_hidden_dim} for nhead={transformer_nhead} before init")
        run_dl_model_pipeline(
            TransformerBaseline, "Transformer",
            {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': current_transformer_hidden_dim, 'nhead': transformer_nhead, 'num_layers': transformer_nlayers},
            args.run_transformer
        )
        run_dl_model_pipeline(
            CNNBaseline, "CNN",
            {'input_dim': input_dim, 'output_dim': output_dim, 'num_channels': cnn_channels, 'kernel_size': cnn_kernel_size},
            args.run_cnn
        )
        run_dl_model_pipeline(
            ResNetBaseline, "ResNet",
            {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': resnet_hidden_dim, 'num_blocks': resnet_blocks},
            args.run_resnet
        )
        run_dl_model_pipeline(
            LSTMBaseline, "LSTM",
            {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': lstm_hidden_dim, 'num_layers': lstm_layers},
            args.run_lstm
        )
        # --- NEW: Run QASA Pipeline ---
        run_dl_model_pipeline(
            QASARegression, "QASA",
            {'input_dim': input_dim, 'output_dim': output_dim, 
             'hidden_dim': qasa_hidden_dim, 'num_layers': qasa_num_layers, 'nhead': qasa_nhead,
             'n_qubits': qasa_n_qubits, 'q_layers': qasa_q_layers}, # Pass correct qubit/layer count
            args.run_qasa
        )
        # --- End QASA Pipeline --- 

        # --- Run Sklearn/Boosting Model Pipelines ---
        if sklearn_available:
            svr_metrics = run_sklearn_pipeline(
                SVR, "SVR",
                {'C': svr_c, 'kernel': args.svr_kernel}, args.run_svr,
                X_train_np, Y_train_np, X_test_np, Y_test_numpy,
                scaler_y, target_names, args.save_dir_prefix
            )
            if svr_metrics: all_metrics["SVR"] = svr_metrics

            knn_metrics = run_sklearn_pipeline(
                KNeighborsRegressor, "KNN",
                {'n_neighbors': args.knn_neighbors}, args.run_knn,
                X_train_np, Y_train_np, X_test_np, Y_test_numpy,
                scaler_y, target_names, args.save_dir_prefix
            )
            if knn_metrics: all_metrics["KNN"] = knn_metrics
        else:
            if args.run_svr: print("\n--- Skipping SVR Model (scikit-learn not available) --- ")
            if args.run_knn: print("\n--- Skipping KNN Model (scikit-learn not available) --- ")


        if xgboost_available:
            xgb_metrics = run_sklearn_pipeline(
                xgb.XGBRegressor, "XGBoost",
                {'n_estimators': args.xgb_n_estimators, 'learning_rate': args.xgb_lr, 'objective': 'reg:squarederror', 'random_state': 42}, args.run_xgboost,
                X_train_np, Y_train_np, X_test_np, Y_test_numpy,
                scaler_y, target_names, args.save_dir_prefix
            )
            if xgb_metrics: all_metrics["XGBoost"] = xgb_metrics
        else:
             if args.run_xgboost: print("\n--- Skipping XGBoost Model (xgboost not available) --- ")

        if lightgbm_available:
            lgbm_metrics = run_sklearn_pipeline(
                lgb.LGBMRegressor, "LightGBM",
                {'n_estimators': args.lgbm_n_estimators, 'learning_rate': args.lgbm_lr, 'random_state': 42}, args.run_lightgbm,
                X_train_np, Y_train_np, X_test_np, Y_test_numpy,
                scaler_y, target_names, args.save_dir_prefix
            )
            if lgbm_metrics: all_metrics["LightGBM"] = lgbm_metrics
        else:
             if args.run_lightgbm: print("\n--- Skipping LightGBM Model (lightgbm not available) --- ")


        # --- Summarize Results ---
        if len(all_metrics) > 1: # Only summarize if more than one model was run
            summarize_results(all_metrics, target_names, args.save_dir_prefix)
        elif len(all_metrics) == 1:
             model_ran = list(all_metrics.keys())[0]
             print(f"\nOnly model '{model_ran}' was run successfully, skipping summary comparison plots.")
        else:
             print("\nNo models were run successfully, skipping summary.")


        print("\nAll specified models processed.")

    except FileNotFoundError:
         print(f"Error: Excel file not found at {args.excel_path}")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 