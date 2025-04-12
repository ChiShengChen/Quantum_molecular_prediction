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
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise schedule as proposed in the improved DDPM paper.
    """
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999).float()

def get_timestep_embedding(timesteps, embedding_dim=128):
    """
    Create sinusoidal time embeddings similar to transformer positional encoding.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=1)

class EnhancedQuantumLayer(nn.Module):
    """
    Improved quantum layer with data re-uploading and trainable rotations
    """
    def __init__(self, n_qubits=24, n_layers=4, input_dim=256):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create a quantum device with more qubits for higher expressivity
        dev = qml.device("default.qubit", wires=n_qubits)
        
        # Define a more complex quantum circuit with data re-uploading
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights_rx, weights_ry, weights_rz, weights_cz, final_rotations):
            # Initial state preparation
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Complex multi-layer variational circuit with data re-uploading
            for l in range(n_layers):
                # Apply rotational gates with trainable parameters
                for i in range(n_qubits):
                    qml.RX(weights_rx[l, i], wires=i)
                    qml.RY(weights_ry[l, i], wires=i)
                    qml.RZ(weights_rz[l, i], wires=i)
                
                # Apply entangling gates - CZ offers different entanglement structure
                for i in range(n_qubits-1):
                    qml.CZ(wires=[i, (i+1) % n_qubits])
                
                # Every other layer, re-upload the data to prevent barren plateaus
                if l % 2 == 1:
                    for i in range(n_qubits):
                        qml.RY(inputs[i] * weights_cz[l, i], wires=i)
            
            # Final rotation layer for measurement preparation
            for i in range(n_qubits):
                qml.RX(final_rotations[i], wires=i)
                
            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Define weight shapes for the quantum circuit
        weight_shapes = {
            "weights_rx": (n_layers, n_qubits),
            "weights_ry": (n_layers, n_qubits),
            "weights_rz": (n_layers, n_qubits),
            "weights_cz": (n_layers, n_qubits),
            "final_rotations": n_qubits
        }
        
        # Create the quantum layer
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
        # Input and output projections with normalization
        # Use Layer Normalization instead of Batch Normalization for better handling of small batches
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, n_qubits),
            nn.LayerNorm(n_qubits)
        )
        
        # Output projection with higher capacity
        self.output_proj = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.SiLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Handle single sample case (just in case)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # Apply input projection
        x = self.input_proj(x)
        
        # Process each sample through the quantum circuit
        results = []
        for sample in x:
            # Move to CPU for quantum processing
            sample_cpu = sample.detach().cpu()
            # Apply quantum circuit
            q_result = self.qlayer(sample_cpu)
            results.append(q_result)
        
        # Stack results and move back to original device
        quantum_output = torch.stack(results).to(x.device)
        
        # Apply output projection
        return self.output_proj(quantum_output)

class SelfAttention(nn.Module):
    """
    Self-attention mechanism for feature interaction
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Adjust input to 3D if needed [batch, seq_len, dim]
        orig_shape = x.shape
        if len(orig_shape) == 2:
            # If input is [batch, dim], add sequence dimension
            x = x.unsqueeze(1)
        
        b, n, c = x.shape
        
        # Check dimensions match
        if c != self.dim:
            raise ValueError(f"Input feature dimension {c} doesn't match expected dimension {self.dim}")
            
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        
        # Return to original shape if needed
        if len(orig_shape) == 2:
            x = x.squeeze(1)
            
        return x

class ImprovedQuantumDiffusionNet(nn.Module):
    """
    Enhanced quantum diffusion network with attention and skip connections
    """
    def __init__(self, input_dim, output_dim=4, hidden_dim=256, use_quantum=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_quantum = use_quantum
        self.debug_counter = 0
        
        # Feature extraction with higher capacity
        self.feature_extract = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Time embedding with higher dimension
        self.time_embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, hidden_dim),
            nn.SiLU()
        )
        
        # Target feature projection
        self.target_proj = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Self-attention layers for feature interaction
        self.cond_attention = SelfAttention(hidden_dim)
        self.post_attention = SelfAttention(hidden_dim)
        
        # Main processing network - fixed input dimension to match hidden_dim
        self.process_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Quantum layer (optional)
        if use_quantum:
            self.quantum_layer = EnhancedQuantumLayer(n_qubits=24, n_layers=4, input_dim=hidden_dim)
        else:
            self.quantum_layer = None
        
        # Output layers with skip connection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, t, cond=None):
        """
        Forward pass through the network
        x: tensor of shape [batch_size, output_dim] - the noisy targets
        t: tensor of shape [batch_size] - the timesteps
        cond: tensor of shape [batch_size, input_dim] - the conditioning features
        """
        # Create debug flag (first few iterations only)
        debug = (self.debug_counter < 5)
        
        # Ensure inputs have correct dimensions
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            if debug:
                print(f"  Unsqueezed x shape: {x.shape}")
                
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
            if debug:
                print(f"  Unsqueezed t shape: {t.shape}")
        
        # Make sure batch dimensions match
        if t.shape[0] != x.shape[0]:
            if t.shape[0] == 1:
                t = t.expand(x.shape[0])
            elif x.shape[0] == 1:
                x = x.expand(t.shape[0], -1)
            if debug:
                print(f"  After batch matching: x shape: {x.shape}, t shape: {t.shape}")
                
        if debug:
            print(f"[DEBUG] ImprovedQuantumDiffusionNet forward: (iter {self.debug_counter})")
            print(f"  x shape: {x.shape}")
            print(f"  t shape: {t.shape}")
            if cond is not None:
                print(f"  cond shape: {cond.shape}")
            else:
                print(f"  cond: None")
        
        # Get batch size
        batch_size = x.shape[0]
        
        # Embed time
        t_emb = get_timestep_embedding(t, embedding_dim=128)
        t_emb = self.time_embed(t_emb)
        
        if debug:
            print(f"  t_emb shape: {t_emb.shape}")
            
        # If there's conditioning, extract features
        if cond is not None:
            # Ensure conditioning has right dimensions
            if len(cond.shape) == 1:
                cond = cond.unsqueeze(0)
                
            # Make sure batch dimensions match
            if cond.shape[0] != x.shape[0]:
                if cond.shape[0] == 1:
                    cond = cond.expand(x.shape[0], -1)
                elif x.shape[0] == 1:
                    x = x.expand(cond.shape[0], -1)
                    t_emb = t_emb.expand(cond.shape[0], -1)
                    
            cond_features = self.feature_extract(cond)
            if debug:
                print(f"  cond_features shape: {cond_features.shape}")
        else:
            # If no conditioning, use zeros
            cond_features = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            if debug:
                print(f"  Created zero cond_features with shape: {cond_features.shape}")
        
        # Project input targets
        h = self.target_proj(x)
        if debug:
            print(f"  After target_proj, h shape: {h.shape}")
        
        # Apply self-attention on target features
        h = h.unsqueeze(1)  # Add sequence dimension for attention
        h = self.cond_attention(h)
        h = h.squeeze(1)    # Remove sequence dimension after attention
        
        if debug:
            print(f"  After cond_attention, h shape: {h.shape}")
        
        # Process through the main network with skip connection
        h_skip = h
        h = self.process_net(h)
        h = h + h_skip
        if debug:
            print(f"  After process_net, h shape: {h.shape}")
        
        # Optional quantum layer
        if self.use_quantum and self.quantum_layer is not None:
            # Apply quantum layer
            h_quantum = self.quantum_layer(h)
            
            # Add skip connection
            h = h + h_quantum
            if debug:
                print(f"  After quantum layer, h shape: {h.shape}")
        
        # Final output layer
        output = self.output_net(h)
        if debug:
            print(f"  Final output shape: {output.shape}")
            
        # Increment debug counter
        self.debug_counter += 1
        
        return output

class AdvancedDiffusionModel:
    """
    Advanced diffusion model with improved sampling
    """
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        # Use cosine schedule for better results
        self.beta = cosine_beta_schedule(timesteps).to(DEVICE)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus = torch.sqrt(1 - self.alpha_bar)
        
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Ensure dimensions match
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
            
        # Add noise according to schedule
        return self.sqrt_alpha_bar[t].view(-1, 1) * x_start + self.sqrt_one_minus[t].view(-1, 1) * noise
    
    @torch.no_grad()
    def p_sample(self, model, x, t, x_cond=None, guidance_scale=1.0):
        """
        Single reverse diffusion step with classifier-free guidance
        """
        # Create debug flag for first few iterations
        debug = True
        
        if debug:
            print(f"\n[DEBUG] p_sample:")
            print(f"  x shape: {x.shape}")
            print(f"  t shape: {t.shape}")
            if x_cond is not None:
                print(f"  x_cond shape: {x_cond.shape}")
            else:
                print(f"  x_cond: None")
        
        # Ensure dimensions match
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            if debug:
                print(f"  x shape after unsqueeze: {x.shape}")
            
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
            if debug:
                print(f"  t shape after unsqueeze: {t.shape}")
            
        # Clamp timestep to valid range
        t = torch.clamp(t, 0, self.timesteps-1)
        
        # Get model prediction with and without conditioning
        if guidance_scale > 1.0 and x_cond is not None:
            try:
                # Unconditional prediction (without conditioning)
                if debug:
                    print(f"  Making unconditional prediction")
                noise_pred_uncond = model(x, t)
                if debug:
                    print(f"  unconditional noise_pred shape: {noise_pred_uncond.shape}")
                
                # Conditional prediction (with conditioning)
                if debug:
                    print(f"  Making conditional prediction")
                noise_pred_cond = model(x, t, x_cond)
                if debug:
                    print(f"  conditional noise_pred shape: {noise_pred_cond.shape}")
                
                # Check that shapes match
                if noise_pred_uncond.shape != noise_pred_cond.shape:
                    print(f"Warning: Shape mismatch in guidance: uncond {noise_pred_uncond.shape}, cond {noise_pred_cond.shape}")
                    # Try to resolve shape mismatch
                    if noise_pred_uncond.shape[0] != noise_pred_cond.shape[0]:
                        # Broadcast to match batch size
                        if noise_pred_uncond.shape[0] == 1:
                            noise_pred_uncond = noise_pred_uncond.expand_as(noise_pred_cond)
                        elif noise_pred_cond.shape[0] == 1:
                            noise_pred_cond = noise_pred_cond.expand_as(noise_pred_uncond)
                
                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                if debug:
                    print(f"  Applied classifier-free guidance with scale: {guidance_scale}")
                    print(f"  Final noise_pred shape: {noise_pred.shape}")
            except Exception as e:
                print(f"Error in classifier-free guidance: {str(e)}, falling back to conditional prediction")
                if debug:
                    print(f"  Error details: {str(e)}")
                    print(f"  Falling back to conditional-only prediction")
                noise_pred = model(x, t, x_cond)
        else:
            # Standard prediction
            if debug:
                print(f"  Making standard prediction (guidance_scale = {guidance_scale})")
            noise_pred = model(x, t, x_cond)
            if debug:
                print(f"  noise_pred shape: {noise_pred.shape}")
        
        # Get variance parameters
        beta_t = self.beta[t].view(-1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        
        if debug:
            print(f"  beta_t shape: {beta_t.shape}")
            print(f"  alpha_bar_t shape: {alpha_bar_t.shape}")
        
        # Handle boundary case for t=0
        if t[0] == 0:
            result = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
            if debug:
                print(f"  t=0 boundary case, result shape: {result.shape}")
            return result
        
        # Compute predicted x_0
        alpha_bar_prev = self.alpha_bar[t - 1].view(-1, 1)
        x0_pred = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        
        if debug:
            print(f"  alpha_bar_prev shape: {alpha_bar_prev.shape}")
            print(f"  x0_pred shape: {x0_pred.shape}")
        
        # Compute mean and variance for the reverse step
        mean = (beta_t * alpha_bar_prev.sqrt() / (1 - alpha_bar_t) * x0_pred +
                (1 - alpha_bar_prev) * self.alpha[t].view(-1, 1).sqrt() / (1 - alpha_bar_t) * x)
        std = (beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt()
        
        # Add noise scaled by standard deviation
        return mean + std * torch.randn_like(x)
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, x_cond=None, guidance_scale=1.0, num_steps=None, verbose=True):
        """
        Full reverse diffusion sampling loop with progressive steps
        """
        debug = verbose  # Use verbose flag to control debugging
        
        if debug:
            print(f"\n[DEBUG] p_sample_loop:")
            print(f"  shape: {shape}")
            if x_cond is not None:
                print(f"  x_cond shape: {x_cond.shape}")
            else:
                print(f"  x_cond: None")
            print(f"  guidance_scale: {guidance_scale}")
            print(f"  num_steps: {num_steps}")
        
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        if debug:
            print(f"  Starting noise x shape: {x.shape}")
        
        # Determine sampling steps
        if num_steps is None:
            num_steps = self.timesteps
            step_indices = list(range(self.timesteps-1, -1, -1))
        else:
            # Use evenly spaced steps
            step_indices = np.linspace(self.timesteps-1, 0, num_steps).round().astype(int)
            
        if debug:
            print(f"  Using {len(step_indices)} steps from {step_indices[0]} to {step_indices[-1]}")
        
        # Create progress bar
        progress = tqdm(step_indices, desc="Sampling", disable=not verbose)
        
        # Progressively denoise
        for i, step in enumerate(progress):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            if debug and (i == 0 or i == len(step_indices) - 1 or i % 50 == 0):
                print(f"  Step {i}/{len(step_indices)} (t={step}): x shape before: {x.shape}")
            
            try:
                x = self.p_sample(model, x, t, x_cond, guidance_scale)
            except Exception as e:
                if debug:
                    print(f"  Error in p_sample at step {i} (t={step}): {str(e)}")
                    import traceback
                    traceback.print_exc()
                    print(f"  x shape: {x.shape}, t shape: {t.shape}")
                    if x_cond is not None:
                        print(f"  x_cond shape: {x_cond.shape}")
                # Try to continue with adjusted parameters if possible
                if "dimension mismatch" in str(e):
                    if debug:
                        print("  Attempting to fix dimension mismatch...")
                    # Try to fix dimensions and continue
                    if len(x.shape) == 1:
                        x = x.unsqueeze(0)
                    if len(t.shape) == 0:
                        t = t.unsqueeze(0)
                    # Try again with fixed dimensions
                    try:
                        x = self.p_sample(model, x, t, x_cond, guidance_scale)
                        if debug:
                            print("  Fixed dimension mismatch successfully")
                    except Exception as e2:
                        if debug:
                            print(f"  Second attempt also failed: {str(e2)}")
                        # Keep x unchanged for this step
                        pass
            
            if debug and (i == 0 or i == len(step_indices) - 1 or i % 50 == 0):
                print(f"  Step {i}/{len(step_indices)} (t={step}): x shape after: {x.shape}")
                # Check for NaN/Inf values
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"  Warning: x contains NaN or Inf values at step {i}")
            
            # Occasionally report progress
            if verbose and i % 20 == 0:
                progress.set_postfix(step=step)
            
        if debug:
            print(f"  Final output shape: {x.shape}")
        
        return x

def load_molecular_data(excel_path, sheet_name, targets):
    """
    Enhanced data loading with advanced preprocessing
    """
    try:
        # Load Excel file
        df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=1)
        print(f"Successfully loaded Excel file with {len(df)} rows")
        
        # Explore data structure
        print(f"Columns found: {len(df.columns)}")
        print(f"Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Check target columns exist
        missing_targets = [t for t in targets if t not in df.columns]
        if missing_targets:
            raise ValueError(f"Target columns not found in data: {missing_targets}")
            
        # Process target columns
        print(f"NaN values in target columns before filtering: {df[targets].isna().sum().sum()}")
        df_clean = df.dropna(subset=targets)
        print(f"Remaining rows after dropping NaN targets: {len(df_clean)}")
        
        # Extract features (numerical columns except targets)
        input_df = df_clean.select_dtypes(include=[float, int]).drop(columns=targets, errors='ignore')
        feature_columns = input_df.columns.tolist()
        
        if len(feature_columns) < 5:
            print("Warning: Very few numerical features found. Check data format.")
            
        print(f"Input features ({len(feature_columns)}): {feature_columns[:5]}...")
        
        # Feature engineering - add interaction terms for top correlated features
        if len(feature_columns) > 5:
            # Calculate correlations with targets
            correlations = []
            for col in feature_columns:
                # Safe correlation calculation
                try:
                    mean_corr = abs(df_clean[col].corr(df_clean[targets].mean(axis=1)))
                    if pd.isna(mean_corr):
                        mean_corr = 0
                except Exception:
                    mean_corr = 0
                correlations.append((col, mean_corr))
            
            # Sort by correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = [c[0] for c in correlations[:min(5, len(correlations))]]
            
            # Add interaction terms
            new_columns = []
            for i in range(len(top_features)):
                for j in range(i+1, len(top_features)):
                    col_name = f"{top_features[i]}_x_{top_features[j]}"
                    # Safe interaction calculation
                    try:
                        input_df[col_name] = df_clean[top_features[i]] * df_clean[top_features[j]]
                        new_columns.append(col_name)
                    except Exception as e:
                        print(f"Warning: Could not create interaction {col_name}: {str(e)}")
            
            print(f"Added {len(new_columns)} interaction terms")
        
        # Handle missing values
        if input_df.isna().any().any():
            print("Imputing missing values with median imputation")
            # Use median imputation (more robust than iterative imputer)
            for col in input_df.columns:
                input_df[col] = input_df[col].fillna(input_df[col].median())
        
        # Extract data arrays
        X, Y = input_df.values, df_clean[targets].values
        
        # Checking for NaN or Inf values
        x_nan_count = np.isnan(X).sum()
        y_nan_count = np.isnan(Y).sum()
        if x_nan_count > 0 or y_nan_count > 0:
            print(f"Warning: Found {x_nan_count} NaN values in X and {y_nan_count} in Y after processing")
        
        # Robust handling of anomalies with quantile-based clipping
        try:
            X = np.clip(X, np.nanquantile(X, 0.001), np.nanquantile(X, 0.999))
            Y = np.clip(Y, np.nanquantile(Y, 0.001), np.nanquantile(Y, 0.999))
        except Exception as e:
            print(f"Warning: Could not apply quantile clipping: {str(e)}")
        
        # Replace remaining NaNs and infinities
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Final data shapes: X: {X.shape}, Y: {Y.shape}")
        
        # Split data with robust stratification
        from sklearn.model_selection import train_test_split
        
        # Create stratification only if enough samples
        if len(X) > 10:
            try:
                # Create quantile-based stratification
                y_mean = np.mean(Y, axis=1)
                n_bins = min(5, len(X) // 5)  # Ensure enough samples per bin
                y_strat = pd.qcut(y_mean, n_bins, labels=False, duplicates='drop')
                stratify = y_strat if len(set(y_strat)) > 1 else None
            except Exception:
                stratify = None
        else:
            stratify = None
        
        # Split with stratification if possible
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        # Use RobustScaler to handle outliers
        scaler_x = RobustScaler()
        scaler_y = RobustScaler()
        
        # Scale the data with error handling
        try:
            X_train_scaled = scaler_x.fit_transform(X_train)
            Y_train_scaled = scaler_y.fit_transform(Y_train)
            X_test_scaled = scaler_x.transform(X_test)
            Y_test_scaled = scaler_y.transform(Y_test)
        except Exception as e:
            print(f"Warning: Error during scaling: {str(e)}. Falling back to StandardScaler")
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)
            Y_train_scaled = scaler_y.fit_transform(Y_train)
            X_test_scaled = scaler_x.transform(X_test)
            Y_test_scaled = scaler_y.transform(Y_test)
        
        # Check for any remaining issues
        for arr, name in [(X_train_scaled, "X_train"), (Y_train_scaled, "Y_train"), 
                          (X_test_scaled, "X_test"), (Y_test_scaled, "Y_test")]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"Warning: Found NaN/Inf in {name} after scaling, replacing with zeros")
                arr = np.nan_to_num(arr)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)
        
        print("Data processing completed successfully")
        
        return (X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, 
                scaler_y, feature_columns, targets)
                
    except Exception as e:
        print(f"Error in data loading: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def train_quantum_diffusion_model(
    X_train, Y_train, X_val, Y_val, 
    input_dim, output_dim, 
    use_quantum=True, 
    epochs=50,
    batch_size=32,
    save_dir="quantum_diffusion_results",
    early_stop_patience=5,
    lr=3e-4,
    timesteps=1000,
    hidden_dim=384,
    guidance_scale=1.5
):
    """
    Train the improved quantum diffusion model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Print shape info for debugging
    print(f"Input dim: {input_dim}, Output dim: {output_dim}, Hidden dim: {hidden_dim}")
    
    # Initialize model and diffusion process
    model = ImprovedQuantumDiffusionNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        use_quantum=use_quantum
    ).to(DEVICE)
    
    # Create EMA model for more stable sampling
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    
    diffusion = AdvancedDiffusionModel(timesteps=timesteps)
    
    # Use AdamW with weight decay and cosine LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Create dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) as tepoch:
            for i, (x, y) in enumerate(tepoch):
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                # Print the first batch shapes for debugging
                if epoch == 0 and i == 0:
                    print(f"Batch x shape: {x.shape}, y shape: {y.shape}")
                
                # Skip if batch contains NaN
                if torch.isnan(x).any() or torch.isnan(y).any():
                    continue
                
                # Sample random timesteps
                t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=DEVICE)
                
                # Add noise to targets
                noise = torch.randn_like(y)
                y_noisy = diffusion.q_sample(y, t, noise)
                
                # Skip if noisy samples contain NaN
                if torch.isnan(y_noisy).any():
                    continue
                
                try:
                    # Predict noise
                    predicted_noise = model(y_noisy, t, x)
                    
                    # Calculate loss (L2 norm)
                    loss = F.mse_loss(predicted_noise, noise)
                    
                    # Regularization losses to reduce NaN/Inf issues
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                        
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Update EMA model
                    with torch.no_grad():
                        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                            p_ema.data = p_ema.data * 0.9995 + p.data * (1 - 0.9995)
                    
                    # Update progress bar
                    tepoch.set_postfix(loss=loss.item())
                    epoch_losses.append(loss.item())
                    
                except Exception as e:
                    # Print detailed error info including tensor shapes
                    print(f"Error in batch {i}, epoch {epoch}:")
                    print(f"  x shape: {x.shape}")
                    print(f"  y shape: {y.shape}")
                    print(f"  y_noisy shape: {y_noisy.shape}")
                    print(f"  t shape: {t.shape}")
                    print(f"  Error: {str(e)}")
                    # Skip this batch
                    continue
        
        # End of epoch processing
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        train_losses.append(avg_train_loss)
        
        # LR scheduler step
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = evaluate_model(model, diffusion, X_val, Y_val, batch_size=batch_size)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
        plt.close()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_ema_state = copy.deepcopy(ema_model.state_dict())
            
            # Save best model
            torch.save({
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss
            }, os.path.join(save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                # Restore best model
                model.load_state_dict(best_model_state)
                ema_model.load_state_dict(best_ema_state)
                break
    
    # Save final model
    torch.save({
        'model': model.state_dict(),
        'ema_model': ema_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1]
    }, os.path.join(save_dir, 'final_model.pt'))
    
    return model, ema_model, diffusion, train_losses, val_losses

def evaluate_model(model, diffusion, X_val, Y_val, batch_size=32):
    """
    Evaluate the model on validation data
    """
    model.eval()
    val_losses = []
    
    # Create dataloader
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (x.shape[0],), device=DEVICE)
            
            # Add noise to targets
            noise = torch.randn_like(y)
            y_noisy = diffusion.q_sample(y, t, noise)
            
            # Predict noise
            predicted_noise = model(y_noisy, t, x)
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            val_losses.append(loss.item())
    
    return sum(val_losses) / len(val_losses) if val_losses else float('inf')

def generate_predictions(ema_model, diffusion, X_test, scaler_y, num_samples=5, num_steps=200, guidance_scale=1.5):
    """
    Generate predictions using the diffusion model
    """
    debug = True
    if debug:
        print(f"\n[DEBUG] generate_predictions:")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  num_samples: {num_samples}")
        print(f"  num_steps: {num_steps}")
        print(f"  guidance_scale: {guidance_scale}")
        print(f"  output_dim: {ema_model.output_dim}")
    
    ema_model.eval()
    all_samples = []
    
    for i, x in enumerate(tqdm(X_test, desc="Generating predictions")):
        # Move to device and add batch dimension
        x = x.to(DEVICE).unsqueeze(0)
        
        if debug and i == 0:
            print(f"  Sample {i}: x shape: {x.shape}")
        
        # Generate multiple samples for uncertainty estimation
        batch_samples = []
        for s in range(num_samples):
            try:
                # Sample from the diffusion model
                shape = (1, ema_model.output_dim)
                if debug and i == 0 and s == 0:
                    print(f"  Sampling with shape: {shape}")
                
                sample = diffusion.p_sample_loop(
                    ema_model, 
                    shape, 
                    x_cond=x, 
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    verbose=(i == 0 and s == 0)  # Only show progress for first sample
                )
                
                # Move to CPU and convert to numpy
                sample_np = sample.cpu().numpy()
                if debug and i == 0 and s == 0:
                    print(f"  Sampled output shape: {sample_np.shape}")
                
                batch_samples.append(sample_np)
            except Exception as e:
                print(f"Error in sampling {i}, sample {s}: {str(e)}")
                if debug:
                    import traceback
                    traceback.print_exc()
                # Use zeros as a fallback
                sample_np = np.zeros((1, ema_model.output_dim))
                batch_samples.append(sample_np)
        
        # Stack samples
        stacked_samples = np.vstack(batch_samples)
        if debug and i == 0:
            print(f"  Stacked samples shape: {stacked_samples.shape}")
            
        all_samples.append(stacked_samples)
    
    # Convert to array of shape [n_test, n_samples, n_features]
    all_samples = np.array(all_samples)
    if debug:
        print(f"  all_samples shape: {all_samples.shape}")
    
    # Inverse transform
    all_samples_denorm = np.zeros_like(all_samples)
    for i in range(len(X_test)):
        for j in range(num_samples):
            try:
                all_samples_denorm[i, j] = scaler_y.inverse_transform(all_samples[i, j].reshape(1, -1)).flatten()
            except Exception as e:
                print(f"Error in inverse transform {i}, sample {j}: {str(e)}")
                # Keep original values as fallback
                all_samples_denorm[i, j] = all_samples[i, j]
    
    if debug:
        print(f"  all_samples_denorm shape: {all_samples_denorm.shape}")
    
    # Calculate mean and std for each prediction
    mean_preds = np.mean(all_samples_denorm, axis=1)
    std_preds = np.std(all_samples_denorm, axis=1)
    
    if debug:
        print(f"  mean_preds shape: {mean_preds.shape}")
        print(f"  std_preds shape: {std_preds.shape}")
    
    return mean_preds, std_preds, all_samples_denorm

def evaluate_predictions(mean_preds, Y_test_denorm, target_names, save_dir="quantum_diffusion_results"):
    """
    Evaluate predictions and create visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare metrics dictionary
    metrics = {target: {} for target in target_names}
    
    # Calculate metrics for each target
    for i, target in enumerate(target_names):
        y_true = Y_test_denorm[:, i]
        y_pred = mean_preds[:, i]
        
        # Remove NaN/Inf values
        valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        if not np.any(valid_idx):
            print(f"Warning: No valid predictions for {target}")
            continue
            
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Save metrics
        metrics[target] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        # Create scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel(f"True {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"{target} - Quantum Diffusion (R² = {r2:.4f})")
        plt.savefig(os.path.join(save_dir, f"scatter_{target}.png"))
        plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({target: pd.Series(target_metrics) 
                              for target, target_metrics in metrics.items()})
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"))
    
    # Save predictions
    pd.DataFrame(mean_preds, columns=target_names).to_csv(
        os.path.join(save_dir, "predictions.csv"), index=False)
    
    return metrics

# Main execution function
def main():
    print("=" * 80)
    print("Quantum Molecular Diffusion Model")
    print("=" * 80)
    
    # Get arguments
    import argparse
    parser = argparse.ArgumentParser(description="Quantum Molecular Diffusion")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (faster training)")
    parser.add_argument("--no-quantum", action="store_true", help="Disable quantum processing")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--hidden-dim", type=int, default=384, help="Hidden dimension size")
    args = parser.parse_args()
    
    # Set debug parameters if needed
    if args.debug:
        print("\nRunning in DEBUG mode - reduced training parameters")
        epochs = 5
        batch_size = 16
        timesteps = 100
        hidden_dim = 256
        guidance_scale = 1.2
        sampling_steps = 50
        patience = 2
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        timesteps = args.timesteps
        hidden_dim = args.hidden_dim
        guidance_scale = 1.5
        sampling_steps = 200
        patience = 5
    
    # Load data
    try:
        excel_path = "pnas.2214357120.sd01.xlsx"
        sheet_name = "Dataset"
        target_columns = ['LCE', 'sC', 'aC', 'C']
        
        print("\nLoading molecular data...")
        (
            X_train, Y_train, X_test, Y_test, 
            scaler_y, feature_names, target_names
        ) = load_molecular_data(excel_path, sheet_name, target_columns)
        
        # Split training data to create validation set
        val_size = int(0.1 * len(X_train))
        X_val, Y_val = X_train[-val_size:], Y_train[-val_size:]
        X_train, Y_train = X_train[:-val_size], Y_train[:-val_size]
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Input dimensions: {X_train.shape[1]}")
        print(f"Output dimensions: {Y_train.shape[1]}")
        
        use_quantum = not args.no_quantum
        model_type = "Quantum" if use_quantum else "Classical"
        
        # Train model (quantum or classical based on argument)
        print(f"\nTraining {model_type} diffusion model...")
        save_dir = f"{model_type.lower()}_diffusion_results"
        model, ema_model, diffusion, train_losses, val_losses = train_quantum_diffusion_model(
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            use_quantum=use_quantum,
            epochs=epochs,
            batch_size=batch_size,
            save_dir=save_dir,
            early_stop_patience=patience,
            lr=3e-4,
            timesteps=timesteps,
            hidden_dim=hidden_dim,
            guidance_scale=guidance_scale
        )
        
        # Generate predictions with uncertainty
        print("\nGenerating predictions with uncertainty quantification...")
        mean_preds, std_preds, all_samples = generate_predictions(
            ema_model=ema_model,
            diffusion=diffusion,
            X_test=X_test,
            scaler_y=scaler_y,
            num_samples=3 if args.debug else 5,
            num_steps=sampling_steps,
            guidance_scale=guidance_scale
        )
        
        # Inverse transform test data
        Y_test_denorm = scaler_y.inverse_transform(Y_test.numpy())
        
        # Evaluate predictions
        print("\nEvaluating predictions...")
        metrics = evaluate_predictions(
            mean_preds=mean_preds,
            Y_test_denorm=Y_test_denorm,
            target_names=target_names,
            save_dir=save_dir
        )
        
        # Print metric summary
        print(f"\n{model_type} model performance metrics:")
        for target, target_metrics in metrics.items():
            print(f"\n{target}:")
            for metric_name, value in target_metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        # If this was the quantum model, also train classical for comparison
        if use_quantum and not args.debug:
            print("\nTraining classical diffusion model (without quantum) for comparison...")
            classical_save_dir = "classical_diffusion_results"
            classical_model, classical_ema, _, _, _ = train_quantum_diffusion_model(
                X_train=X_train,
                Y_train=Y_train,
                X_val=X_val,
                Y_val=Y_val,
                input_dim=X_train.shape[1],
                output_dim=Y_train.shape[1],
                use_quantum=False,  # No quantum layer
                epochs=epochs,
                batch_size=batch_size,
                save_dir=classical_save_dir,
                early_stop_patience=patience,
                lr=3e-4,
                timesteps=timesteps,
                hidden_dim=hidden_dim,
                guidance_scale=guidance_scale
            )
            
            # Generate predictions with classical model
            print("\nGenerating predictions with classical model...")
            classical_mean_preds, classical_std_preds, classical_samples = generate_predictions(
                ema_model=classical_ema,
                diffusion=diffusion,
                X_test=X_test,
                scaler_y=scaler_y,
                num_samples=5,
                num_steps=sampling_steps,
                guidance_scale=guidance_scale
            )
            
            # Evaluate classical predictions
            print("\nEvaluating classical predictions...")
            classical_metrics = evaluate_predictions(
                mean_preds=classical_mean_preds,
                Y_test_denorm=Y_test_denorm,
                target_names=target_names,
                save_dir=classical_save_dir
            )
            
            # Compare models
            print("\nComparing quantum vs classical models:")
            for target in target_names:
                quantum_r2 = metrics[target]['R2']
                classical_r2 = classical_metrics[target]['R2']
                print(f"{target}: Quantum R² = {quantum_r2:.4f}, Classical R² = {classical_r2:.4f}, " +
                      f"Improvement: {100 * (quantum_r2 - classical_r2) / abs(classical_r2):.2f}%")
        
        print(f"\n{model_type} Molecular Diffusion completed successfully!")
        
    except Exception as e:
        print(f"\nError in execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 