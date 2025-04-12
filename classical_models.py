import torch
import torch.nn as nn
import torch.nn.init as init

# Try importing the helper from quantum_models
try:
    from quantum_models import init_weights
except ImportError:
    # Fallback definition if import fails (e.g., running file standalone)
    print("Warning: Could not import init_weights from quantum_models. Using local definition.")
    def init_weights(m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
             init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
             if m.bias is not None:
                  init.constant_(m.bias, 0)

print("Initializing Classical Models Module...")

# --- Baseline Models ---
class MLPBaseline(nn.Module):
    """ Simple MLP baseline. """
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)

class TransformerBaseline(nn.Module):
    """ Simple Transformer Encoder baseline. """
    def __init__(self, input_dim, output_dim, hidden_dim=128, nhead=4, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Ensure d_model is divisible by nhead
        compatible_hidden_dim = hidden_dim
        if hidden_dim % nhead != 0:
            compatible_hidden_dim = (hidden_dim // nhead) * nhead
            if compatible_hidden_dim == 0: compatible_hidden_dim = nhead
            print(f"Adjusting Transformer hidden_dim {hidden_dim} to {compatible_hidden_dim} for nhead={nhead}")
            self.input_proj = nn.Linear(input_dim, compatible_hidden_dim) # Re-init if changed
        else:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=compatible_hidden_dim, 
            nhead=nhead, 
            batch_first=True, 
            dim_feedforward=compatible_hidden_dim*2,
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(compatible_hidden_dim, output_dim)
        self.hidden_dim_adjusted = compatible_hidden_dim # Store adjusted dim if needed
        # Apply init
        self.input_proj.apply(init_weights)
        self.output.apply(init_weights)

    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        # Add sequence dimension if needed
        if x.dim() == 2:
             x = x.unsqueeze(1) # Shape: [batch, seq_len=1, features]
        # Pass through transformer
        x = self.transformer(x) # Shape: [batch, seq_len=1, features]
        # Take output corresponding to the sequence element
        if x.dim() == 3:
             x = x[:, 0, :] # Shape: [batch, features]
        return self.output(x)

class CNNBaseline(nn.Module):
    """ Simple 1D CNN baseline. """
    def __init__(self, input_dim, output_dim, num_channels=[32, 64], kernel_size=3, dropout_rate=0.1):
        super().__init__()
        layers = []
        in_channels = 1 # Treat input features as a sequence with 1 channel
        current_dim = input_dim

        for i, out_channels in enumerate(num_channels):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2)))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels
            
        # Use Adaptive Avg Pooling to always get a fixed size output (1) regardless of input length changes
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten()) # Flatten the output for the linear layer
        self.conv_layers = nn.Sequential(*layers)
        
        # The size fed into the final FC layer is the number of channels from the last conv layer
        flattened_size = num_channels[-1]
        self.fc = nn.Linear(flattened_size, output_dim)
        # Apply init
        self.conv_layers.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        # Input x shape: [batch_size, input_dim]
        # Reshape to [batch_size, channels=1, sequence_length=input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return self.fc(x)

class ResNetBlock(nn.Module):
    """ Basic Residual Block for MLP/ResNet. """
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        # Apply init
        self.layer1.apply(init_weights)
        self.layer2.apply(init_weights)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.norm2(out)
        out += identity # Skip connection
        return self.relu(out)

class ResNetBaseline(nn.Module):
    """ Simple ResNet baseline. """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.1):
        super().__init__()
        self.input_layer = nn.Sequential( # Add norm/activation to input layer too
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Apply init
        self.input_layer.apply(init_weights)
        # Blocks apply init internally
        self.output_layer.apply(init_weights)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)

class LSTMBaseline(nn.Module):
    """ Simple LSTM baseline. """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM expects input shape: (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size=input_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=num_layers, 
                              batch_first=True,
                              dropout=dropout_rate if num_layers > 1 else 0) # Use batch_first=True
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Apply init
        self.fc.apply(init_weights)

    def forward(self, x):
        # Input x shape: [batch_size, input_dim]
        # Reshape to [batch_size, seq_len=1, input_size=input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Forward propagate LSTM (use default zero initial states)
        out, _ = self.lstm(x) # out: tensor of shape (batch_size, seq_length=1, hidden_size)
        
        # Decode the hidden state of the last time step
        # Since seq_len is 1 and batch_first=True, out[:, -1, :] gets the last step's output
        out = self.fc(out[:, -1, :]) 
        return out 