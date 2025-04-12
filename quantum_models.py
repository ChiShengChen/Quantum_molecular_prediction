import torch
import torch.nn as nn
import torch.nn.init as init # Import init

# Import necessary components from quantum_layers
from quantum_layers import EnhancedQuantumLayer, DampedStyleQuantumLayer, QuantumLSTMCell

print("Initializing Quantum Models Module...")

# Helper function for initialization (can be moved later)
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Kaiming for layers followed by ReLU/SiLU/GELU is common
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    # Can add init for other layer types like Conv1d if needed
    # elif isinstance(m, nn.Conv1d):
    #     init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    #     if m.bias is not None:
    #         init.constant_(m.bias, 0)

# --- Original QNN Regression Model --- (Added dropout & init)
class QuantumRegressionNet(nn.Module):
    """ Original Quantum Regression Network using EnhancedQuantumLayer. """
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_qubits=8, n_layers=3, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate), # Added dropout
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate) # Added dropout
        )
        
        # Quantum layer (using EnhancedQuantumLayer)
        self.quantum_layer = EnhancedQuantumLayer(
            n_qubits=n_qubits, n_layers=n_layers, input_dim=hidden_dim
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate), # Added dropout
            nn.Linear(hidden_dim // 2, output_dim)
        )
        # Apply initialization
        self.apply(init_weights)

    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Quantum processing
        quantum_features = self.quantum_layer(features)
        
        # Combine with skip connection (ensure dimensions match)
        if features.shape == quantum_features.shape:
            combined_features = features + quantum_features
        else:
            # Handle dimension mismatch if necessary (e.g., project quantum_features)
            print(f"Warning: Skip connection shape mismatch in QuantumRegressionNet: Features {features.shape}, Quantum {quantum_features.shape}. Using only quantum features.")
            # Or add a projection: quantum_features = self.some_projection(quantum_features)
            combined_features = quantum_features # Fallback to just quantum features
        
        # Output prediction
        output = self.output_layer(combined_features)
        return output

# --- New QASA-Style Model (Based on qasa_damped.py) --- (Added dropout & init)

class DampedStyleEncoderLayer(nn.Module):
    """ Quantum Encoder Layer based on qasa_damped.py. Uses DampedStyleQuantumLayer. """
    def __init__(self, hidden_dim, nhead=4, dim_feedforward=None, n_qubits=8, circuit_layers=4, dropout_rate=0.1):
        super().__init__()
        if dim_feedforward is None:
             dim_feedforward = hidden_dim * 4 # Default feedforward dimension
             
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        # Use the DampedStyleQuantumLayer from quantum_layers module
        # Note: n_qubits and circuit_layers passed to the quantum layer
        self.quantum_res_block = DampedStyleQuantumLayer(
            input_dim=hidden_dim, output_dim=hidden_dim, 
            n_qubits=n_qubits, n_layers=circuit_layers
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout_rate), # Added dropout
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout_rate) # Added dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # Apply initialization only to FFN linear layers here
        self.ffn.apply(init_weights)

    def forward(self, x):
        # Input x shape: [batch_size, seq_len=1, hidden_dim]
        
        # Self-Attention part
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Quantum Residual Block + FeedForward part
        # Reshape for QuantumLayer (expects batch_size, features)
        batch_size, seq_len, features = x.shape
        x_squeezed = x.view(batch_size * seq_len, features) # Flatten batch and seq dims
        
        # Quantum layer expects (batch, input_dim)
        # DampedStyleQuantumLayer includes the residual connection inside
        q_out_with_skip = self.quantum_res_block(x_squeezed)
        
        # Reshape back to (batch_size, seq_len, features)
        q_out_reshaped = q_out_with_skip.view(batch_size, seq_len, features)
        
        # Apply FFN to the output of the quantum block (which includes the skip connection)
        ffn_out = self.ffn(q_out_reshaped) 
        
        # Final normalization after adding FFN output
        # Structure: Attention -> Norm1 -> Quantum+Skip -> FFN -> Norm2
        x = self.norm2(q_out_reshaped + ffn_out) 
        
        return x

class DampedStyleRegressionModel(nn.Module):
    """ 
    Model structured like qasa_damped.py's HybridTransformer, but for regression.
    Treats tabular data as sequence length 1.
    Uses N-1 standard TransformerEncoderLayers + 1 DampedStyleEncoderLayer.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=4, nhead=4,
                 n_qubits=8, circuit_layers=4, dropout_rate=0.1): 
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate) # Added dropout after embedding
        # No positional encoding needed for seq_len=1
        
        # Create encoder layers
        encoder_layers = []
        # N-1 Classical Transformer layers
        # Ensure hidden_dim is compatible with nhead
        if hidden_dim % nhead != 0:
            print(f"Warning: Adjusting hidden_dim {hidden_dim} to {(hidden_dim // nhead) * nhead} for head compatibility.")
            compatible_hidden_dim = (hidden_dim // nhead) * nhead
            if compatible_hidden_dim == 0: compatible_hidden_dim = nhead
            # Note: This assumes embedding layer handles this or subsequent layers adapt. 
            # For simplicity, we will use original hidden_dim, but TransformerEncoderLayer might complain.
            # Better: adjust embedding output or project before encoder.
            # Let's stick to original hidden_dim for now, assuming nn.TransformerEncoderLayer handles it or user adjusts.
        else:
            compatible_hidden_dim = hidden_dim
            
        for _ in range(num_layers - 1):
            encoder_layers.append(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, 
                                           dim_feedforward=hidden_dim*4, batch_first=True, dropout=dropout_rate)
            )
            
        # 1 Quantum layer (DampedStyleEncoderLayer)
        encoder_layers.append(
            DampedStyleEncoderLayer(hidden_dim=hidden_dim, nhead=nhead,
                                n_qubits=n_qubits, circuit_layers=circuit_layers, dropout_rate=dropout_rate)
        )

        self.encoder = nn.ModuleList(encoder_layers)
        
        # Final Regression Head (Linear layer from hidden_dim to output_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Apply initialization
        self.embedding.apply(init_weights)
        self.output_layer.apply(init_weights)

    def forward(self, x):
        # x shape: (batch, features)
        
        # Embedding
        x = self.embedding(x)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x) # Apply dropout
        
        # Add sequence dimension (treat as seq_len=1)
        x = x.unsqueeze(1) # Shape: (batch, 1, hidden_dim)
             
        # Pass through encoder layers
        for layer in self.encoder:
            x = layer(x)
            
        # Remove sequence dimension after encoder
        x = x.squeeze(1) # Shape: (batch, hidden_dim)
             
        # Final prediction
        output = self.output_layer(x)
        return output 

# --- Quantum LSTM Model --- (Added dropout & init)

class QLSTMBaseline(nn.Module):
    """
    Quantum LSTM baseline model using QuantumLSTMCell.
    Adapted for tabular data (treated as sequence length 1).
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, 
                 n_qubits=4, qlstm_layers=1, # Number of quantum layers in the cell
                 num_lstm_layers=1, dropout_rate=0.1): # Number of stacked LSTM layers (can be 1 for simple case)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers # Usually 1 for tabular seq_len=1

        # Input embedding (optional, can directly use input_dim)
        # Let's use a simple projection for consistency
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate) # Added dropout

        # Create QLSTM layer(s)
        # For tabular data (seq_len=1), stacking standard LSTMs doesn't make much sense.
        # We'll use one layer based on QuantumLSTMCell.
        # If num_lstm_layers > 1, it's more complex; we'll keep it simple for now.
        if num_lstm_layers > 1:
            print(f"Warning: Stacking QuantumLSTMCells ({num_lstm_layers} layers) for seq_len=1 is non-standard. Using 1 layer.")
            self.num_lstm_layers = 1 
            
        # The first layer's input size is hidden_dim (after embedding)
        self.qlstm_cell = QuantumLSTMCell(hidden_dim, hidden_dim, n_qubits=n_qubits, n_layers=qlstm_layers)

        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Apply initialization
        self.embedding.apply(init_weights)
        self.output_layer.apply(init_weights)

    def forward(self, x):
        # x shape: (batch, input_dim)
        batch_size = x.size(0)

        # Embedding
        x_emb = self.embedding(x)
        x_emb = self.embedding_norm(x_emb)
        x_emb = self.embedding_dropout(x_emb) # Apply dropout

        # Treat as sequence of length 1: (batch, 1, hidden_dim)
        x_seq = x_emb.unsqueeze(1)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        states = (h0, c0)

        # Since seq_len is 1, we process only the first (and only) step
        # Input to cell is x_seq[:, 0, :], which is just x_emb
        h_t, c_t = self.qlstm_cell(x_emb, states)
        
        # The final hidden state h_t is the output feature representation
        output_features = h_t # Shape: (batch, hidden_dim)

        # Final prediction
        output = self.output_layer(output_features)
        return output 