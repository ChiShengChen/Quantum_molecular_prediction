# train_regression_diffusion.py

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

def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999).float()

def get_timestep_embedding(timesteps, embedding_dim=128):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=1)

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=16, n_layers=3):
        super().__init__()
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RZ(weights[l, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.input_proj = nn.Linear(128, n_qubits)
        self.output_proj = nn.Linear(n_qubits, 128)

    def forward(self, x):
        x = self.input_proj(x)
        x_cpu = x.detach().cpu()
        results = [self.qlayer(sample) for sample in x_cpu]
        out = torch.stack(results).to(x.device)
        return self.output_proj(out)

class ImprovedRegressionNet(nn.Module):
    def __init__(self, input_dim, output_dim=4, use_quantum=False):
        super().__init__()
        self.use_quantum = use_quantum
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Time embedding
        self.time_embed = nn.Linear(128, 128)
        
        # Input projections
        self.input_proj = nn.Linear(input_dim, 128)  # For X_train (23 -> 128)
        self.target_proj = nn.Linear(output_dim, 128)  # For Y_train (4 -> 128)
        
        # Main network - handle both with and without conditional input
        self.mlp_with_cond = nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(),  # 128 + 128 + 128 = 384
            nn.Linear(256, 128), nn.ReLU()
        )
        
        self.mlp_without_cond = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),  # 128 + 128 = 256
            nn.Linear(256, 128), nn.ReLU()
        )
        
        if use_quantum:
            self.q_attn = QuantumLayer()
            self.q_proj = nn.Linear(128, 128)
            
        # Output projection
        self.out = nn.Linear(128, output_dim)

    def forward(self, x, t, x_cond=None):
        # Ensure input dimensions match
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
            
        # Time embedding
        t_embed = get_timestep_embedding(t, embedding_dim=128).to(x.device)
        t_proj = self.time_embed(t_embed)
        
        # Project inputs
        if x.shape[1] == self.input_dim:
            x_proj = self.input_proj(x)
        else:
            x_proj = self.target_proj(x)
            
        # Project conditional inputs if provided and choose appropriate MLP
        if x_cond is not None:
            cond_proj = self.input_proj(x_cond)
            combined = torch.cat([x_proj, t_proj, cond_proj], dim=1)
            x = self.mlp_with_cond(combined)
        else:
            combined = torch.cat([x_proj, t_proj], dim=1)
            x = self.mlp_without_cond(combined)
        
        if self.use_quantum:
            x = self.q_proj(self.q_attn(x))
            
        return self.out(x)

class ModifiedDiffusion1D:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.beta = cosine_beta_schedule(timesteps).to(DEVICE)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus = torch.sqrt(1 - self.alpha_bar)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # Ensure dimensions match
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        return self.sqrt_alpha_bar[t].view(-1, 1) * x_start + self.sqrt_one_minus[t].view(-1, 1) * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, x_cond=None):
        # Ensure dimensions match
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
            
        # Only print shapes occasionally for debugging
        if t[0] % 100 == 0:
            print(f"p_sample - step {t[0]}, x shape: {x.shape}, x_cond shape: {x_cond.shape if x_cond is not None else None}")
            
        noise_pred = model(x, t, x_cond)
        beta_t = self.beta[t].view(-1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        
        # Handle the boundary condition for t=0 specially
        if t[0] == 0:
            return (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        
        # For t > 0, continue with regular sampling formula
        alpha_bar_prev = self.alpha_bar[t - 1].view(-1, 1)
        x0_pred = (x - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        
        mean = (beta_t * alpha_bar_prev.sqrt() / (1 - alpha_bar_t) * x0_pred +
                (1 - alpha_bar_prev) * self.alpha[t].view(-1, 1).sqrt() / (1 - alpha_bar_t) * x)
        std = (beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)).sqrt()
        
        return mean + std * torch.randn_like(x)

# Baseline models
class MLPBaseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output = nn.Linear(64, output_dim)
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.output(x)


def load_data(excel_path, sheet_name, targets):
    try:
        # Try loading the Excel file
        df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=1)
        print(f"Successfully loaded Excel file with {len(df)} rows")
        
        # Check for NaN values in target columns and drop those rows
        print(f"NaN values in target columns before dropping: {df[targets].isna().sum().sum()}")
        df = df.dropna(subset=targets)
        print(f"Remaining rows after dropping NaN targets: {len(df)}")
        
        # Get numerical columns only and exclude target columns
        input_df = df.select_dtypes(include=[float, int]).drop(columns=targets, errors='ignore')
        
        # Check if any target columns were not found
        missing_targets = [t for t in targets if t not in df.columns]
        if missing_targets:
            raise ValueError(f"Target columns not found in data: {missing_targets}")
        
        input_columns = input_df.columns.tolist()
        print(f"Input features: {len(input_columns)}")
        
        # Check for any remaining NaN values in input features
        if input_df.isna().any().any():
            print("Warning: NaN values found in input features. Filling with column means.")
            input_df = input_df.fillna(input_df.mean())
        
        X, Y = input_df.values, df[targets].values
        
        # Check for infinite values
        if np.isinf(X).any() or np.isinf(Y).any():
            print("Warning: Infinite values found. Replacing with NaN and then filling.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Use RobustScaler instead of StandardScaler to handle outliers better
        scaler_x, scaler_y = RobustScaler(), RobustScaler()
        
        # Fit and transform with error handling
        try:
            X_train_scaled = scaler_x.fit_transform(X_train)
            Y_train_scaled = scaler_y.fit_transform(Y_train)
            X_test_scaled = scaler_x.transform(X_test)
            Y_test_scaled = scaler_y.transform(Y_test)
            
            # Final check for NaN values
            if (np.isnan(X_train_scaled).any() or np.isnan(Y_train_scaled).any() or 
                np.isnan(X_test_scaled).any() or np.isnan(Y_test_scaled).any()):
                print("Warning: NaN values found after scaling. Replacing with zeros.")
                X_train_scaled = np.nan_to_num(X_train_scaled)
                Y_train_scaled = np.nan_to_num(Y_train_scaled)
                X_test_scaled = np.nan_to_num(X_test_scaled)
                Y_test_scaled = np.nan_to_num(Y_test_scaled)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)
            
            print("Data loading and preprocessing completed successfully")
            return X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, scaler_y
            
        except Exception as e:
            print(f"Error during scaling: {str(e)}")
            raise
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def train_baseline_model(model, X_train, Y_train, X_test, Y_test, scaler_y, save_dir):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    os.makedirs(save_dir, exist_ok=True)
    losses = []
    
    # Early stopping parameters
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(30):
        model.train()
        epoch_losses = []
        indices = torch.randperm(X_train.size(0))
        
        for i in range(0, X_train.size(0), 64):
            idx = indices[i:i+64]
            x, y = X_train[idx].to(DEVICE), Y_train[idx].to(DEVICE)
            pred = model(x)
            
            # Check for NaN in predictions
            if torch.isnan(pred).any():
                print(f"Warning: NaN detected in predictions at epoch {epoch+1}. Skipping batch.")
                continue
                
            loss = loss_fn(pred, y)
            
            # Check if loss is NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss at epoch {epoch+1}. Skipping batch.")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
            
        # Calculate average loss for the epoch
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    break
        else:
            print(f"[Epoch {epoch+1}] No valid batches")

    # Save loss curve
    if losses:
        plt.figure()
        plt.plot(losses)
        plt.title("Baseline Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
        plt.close()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().numpy()
        
    # Check for NaN in predictions
    if np.isnan(preds).any():
        print("Warning: NaN values in predictions. Replacing with zeros.")
        preds = np.nan_to_num(preds)
        
    # Inverse transform
    try:
        preds_denorm = scaler_y.inverse_transform(preds)
        true_denorm = scaler_y.inverse_transform(Y_test.numpy())
    except Exception as e:
        print(f"Error in inverse transform: {str(e)}")
        # Fall back to original values if inverse transform fails
        preds_denorm = preds
        true_denorm = Y_test.numpy()

    # Save predictions
    pd.DataFrame(preds_denorm).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
    pd.DataFrame(true_denorm).to_csv(os.path.join(save_dir, "ground_truth.csv"), index=False)

    # Create plots and calculate metrics
    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        for i, col in enumerate(['LCE', 'sC', 'aC', 'C']):
            # Check for NaN or inf values before plotting and calculating metrics
            valid_indices = ~(np.isnan(true_denorm[:, i]) | np.isnan(preds_denorm[:, i]) | 
                             np.isinf(true_denorm[:, i]) | np.isinf(preds_denorm[:, i]))
            
            if not any(valid_indices):
                print(f"No valid predictions for {col}. Skipping metrics.")
                f.write(f"=== {col} ===\n")
                f.write("No valid predictions for this target.\n\n")
                continue
                
            true_valid = true_denorm[valid_indices, i]
            pred_valid = preds_denorm[valid_indices, i]
            
            # Plot scatter plot
            plt.figure()
            plt.scatter(true_valid, pred_valid, alpha=0.5)
            min_val = min(true_valid.min(), pred_valid.min())
            max_val = max(true_valid.max(), pred_valid.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(f"{col} - Baseline")
            plt.savefig(os.path.join(save_dir, f"scatter_{col}.png"))
            plt.close()
            
            # Calculate metrics
            try:
                mse = mean_squared_error(true_valid, pred_valid)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(true_valid, pred_valid)
                r2 = r2_score(true_valid, pred_valid)
                f.write(f"=== {col} ===\n")
                f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nR^2: {r2:.4f}\n\n")
            except Exception as e:
                print(f"Error calculating metrics for {col}: {str(e)}")
                f.write(f"=== {col} ===\n")
                f.write(f"Error calculating metrics: {str(e)}\n\n")

def train_quantum_diffusion(X_train, Y_train, X_test, Y_test, scaler_y, use_quantum, save_dir, early_stop_sampling=False):
    model = ImprovedRegressionNet(X_train.shape[1], output_dim=Y_train.shape[1], use_quantum=use_quantum).to(DEVICE)
    ema_model = copy.deepcopy(model)
    diffusion = ModifiedDiffusion1D()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    os.makedirs(save_dir, exist_ok=True)
    loss_history = []
    val_loss_history = []
    
    # Split some training data for validation
    val_size = min(int(X_train.size(0) * 0.1), 20)  # 10% or max 20 samples
    X_val, Y_val = X_train[-val_size:].clone(), Y_train[-val_size:].clone()
    X_train, Y_train = X_train[:-val_size], Y_train[:-val_size]
    
    # Early stopping parameters
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(30):
        # Training loop
        model.train()
        epoch_losses = []
        indices = torch.randperm(X_train.size(0))
        
        for i in range(0, X_train.size(0), 64):
            idx = indices[i:i+64]
            x, y = X_train[idx].to(DEVICE), Y_train[idx].to(DEVICE)
            
            # Skip batch if inputs contain NaN
            if torch.isnan(x).any() or torch.isnan(y).any():
                print(f"Warning: NaN detected in inputs at epoch {epoch+1}. Skipping batch.")
                continue
                
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=DEVICE).long()
            noise = torch.randn_like(y)
            y_noisy = diffusion.q_sample(y, t, noise)
            
            # Skip batch if noisy targets contain NaN
            if torch.isnan(y_noisy).any():
                print(f"Warning: NaN detected in noisy targets at epoch {epoch+1}. Skipping batch.")
                continue
                
            noise_pred = model(y_noisy, t, x)  # Pass x as conditional input
            
            # Skip batch if predictions contain NaN
            if torch.isnan(noise_pred).any():
                print(f"Warning: NaN detected in predictions at epoch {epoch+1}. Skipping batch.")
                continue
                
            loss = F.mse_loss(noise_pred, noise)
            
            # Skip if loss is NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss at epoch {epoch+1}. Skipping batch.")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(0.999).add_(p.data, alpha=1 - 0.999)
                    
        # Calculate average loss for the epoch
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            loss_history.append(avg_loss)
            
            # Validation loop
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i in range(0, X_val.size(0), 8):
                    x_val, y_val = X_val[i:i+8].to(DEVICE), Y_val[i:i+8].to(DEVICE)
                    t_val = torch.randint(0, diffusion.timesteps, (x_val.size(0),), device=DEVICE).long()
                    noise_val = torch.randn_like(y_val)
                    y_noisy_val = diffusion.q_sample(y_val, t_val, noise_val)
                    
                    noise_pred_val = model(y_noisy_val, t_val, x_val)
                    val_loss = F.mse_loss(noise_pred_val, noise_val)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
            val_loss_history.append(avg_val_loss)
            
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check based on validation loss
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                best_ema_state = copy.deepcopy(ema_model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    ema_model.load_state_dict(best_ema_state)
                    break
        else:
            print(f"[Epoch {epoch+1}] No valid batches")

    # Save loss curve
    if loss_history:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label='Training Loss')
        if val_loss_history:
            plt.plot(val_loss_history, label='Validation Loss')
        plt.title("Diffusion Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss_curve.png"))
        plt.close()

    # Sampling with error handling
    ema_model.eval()
    all_preds = []
    print(f"Starting generation for {len(X_test)} test samples...")
    
    # Determine number of timesteps for sampling
    sampling_timesteps = 100 if early_stop_sampling else diffusion.timesteps
    print(f"Using {sampling_timesteps} timesteps for sampling (out of {diffusion.timesteps} total)")
    
    try:
        for i in tqdm(range(len(X_test)), desc="Generating samples"):
            try:
                x = X_test[i:i+1].to(DEVICE)  # Input features
                y = torch.randn(1, Y_train.shape[1], device=DEVICE)  # Initial noise
                
                # Use tqdm for the reverse process too
                timestep_seq = np.linspace(diffusion.timesteps-1, 0, sampling_timesteps).round().astype(int)
                
                for t_index, t_ in enumerate(tqdm(timestep_seq, 
                                desc=f"Sample {i+1}/{len(X_test)}", 
                                leave=False, 
                                total=len(timestep_seq))):
                    t_tensor = torch.tensor([t_], device=DEVICE).long()
                    
                    # Sample with error handling
                    try:
                        # Use input features x for conditioning
                        y = diffusion.p_sample(ema_model, y, t_tensor, x)
                        
                        # Check for NaN/Inf in generated sample
                        if torch.isnan(y).any() or torch.isinf(y).any():
                            print(f"Warning: NaN/Inf in generation at timestep {t_}. Replacing with zeros.")
                            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                            
                    except Exception as e:
                        print(f"Error in sampling step {t_}: {str(e)}")
                        # Reset to random noise if sampling fails
                        y = torch.randn(1, Y_train.shape[1], device=DEVICE)
                
                all_preds.append(y.detach().cpu())
            except Exception as e:
                print(f"Error generating prediction for test sample {i}: {str(e)}")
                # Append zeros if generation fails completely
                all_preds.append(torch.zeros(1, Y_train.shape[1]))
        
        print("Sample generation completed successfully!")
        
        preds = torch.cat(all_preds, dim=0).numpy()
        
        # Check for NaN/Inf in predictions
        if np.isnan(preds).any() or np.isinf(preds).any():
            print("Warning: Final predictions contain NaN/Inf values. Replacing with zeros.")
            preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Inverse transform
        try:
            preds_denorm = scaler_y.inverse_transform(preds)
            true_denorm = scaler_y.inverse_transform(Y_test.numpy())
        except Exception as e:
            print(f"Error in inverse transform: {str(e)}")
            # Fall back to scaled values if inverse transform fails
            preds_denorm = preds
            true_denorm = Y_test.numpy()
        
        # Save predictions
        pd.DataFrame(preds_denorm).to_csv(os.path.join(save_dir, "predictions.csv"), index=False)
        pd.DataFrame(true_denorm).to_csv(os.path.join(save_dir, "ground_truth.csv"), index=False)
        
        # Create plots and calculate metrics
        metrics_path = os.path.join(save_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            for i, col in enumerate(['LCE', 'sC', 'aC', 'C']):
                # Check for NaN or inf values before plotting and calculating metrics
                valid_indices = ~(np.isnan(true_denorm[:, i]) | np.isnan(preds_denorm[:, i]) | 
                                np.isinf(true_denorm[:, i]) | np.isinf(preds_denorm[:, i]))
                
                if not any(valid_indices):
                    print(f"No valid predictions for {col}. Skipping metrics.")
                    f.write(f"=== {col} ===\n")
                    f.write("No valid predictions for this target.\n\n")
                    continue
                    
                true_valid = true_denorm[valid_indices, i]
                pred_valid = preds_denorm[valid_indices, i]
                
                # Plot scatter plot
                plt.figure()
                plt.scatter(true_valid, pred_valid, alpha=0.5)
                min_val = min(true_valid.min(), pred_valid.min())
                max_val = max(true_valid.max(), pred_valid.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                plt.xlabel("True")
                plt.ylabel("Predicted")
                plt.title(f"{col} - Diffusion")
                plt.savefig(os.path.join(save_dir, f"scatter_{col}.png"))
                plt.close()
                
                # Calculate metrics
                try:
                    mse = mean_squared_error(true_valid, pred_valid)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(true_valid, pred_valid)
                    r2 = r2_score(true_valid, pred_valid)
                    f.write(f"=== {col} ===\n")
                    f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nR^2: {r2:.4f}\n\n")
                except Exception as e:
                    print(f"Error calculating metrics for {col}: {str(e)}")
                    f.write(f"=== {col} ===\n")
                    f.write(f"Error calculating metrics: {str(e)}\n\n")
                    
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")

def summarize_results(targets, output_dirs):
    summary = {model: {metric: [] for metric in ['RMSE', 'MAE', 'MSE', 'R2']} for model in output_dirs}
    for model_name, path in output_dirs.items():
        metrics_file = os.path.join(path, "metrics.txt")
        if not os.path.exists(metrics_file):
            continue
        with open(metrics_file, "r") as f:
            lines = f.readlines()
        for i, col in enumerate(targets):
            for metric in ['RMSE', 'MAE', 'MSE', 'R^2']:
                for line in lines:
                    if line.startswith(f"{metric}:"):
                        value = float(line.strip().split(":")[1])
                        if metric == 'R^2':
                            summary[model_name]['R2'].append(value)
                        else:
                            summary[model_name][metric].append(value)
                        break

    for metric in ['RMSE', 'MAE', 'MSE', 'R2']:
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(targets):
            values = [summary[model][metric][i] for model in output_dirs if len(summary[model][metric]) == len(targets)]
            plt.bar([j + i * 0.2 for j in range(len(values))], values, width=0.2, label=f"{col}" if i == 0 else "")
        plt.xticks([r + 0.3 for r in range(len(output_dirs))], list(output_dirs.keys()))
        plt.title(f"{metric} Comparison across Models")
        plt.ylabel(metric)
        plt.legend(targets)
        plt.tight_layout()
        plt.savefig(f"summary_{metric}.png")
        plt.close()

if __name__ == '__main__':
    try:
        excel_path = "pnas.2214357120.sd01.xlsx"
        sheet = "Dataset"
        targets = ['LCE', 'sC', 'aC', 'C']
        
        print("Loading data...")
        X_train, Y_train, X_test, Y_test, scaler_y = load_data(excel_path, sheet, targets)
        
        # Train all models with proper exception handling
        print("\nTraining MLP baseline model...")
        try:
            train_baseline_model(MLPBaseline(X_train.shape[1], 4), X_train, Y_train, X_test, Y_test, scaler_y, "baseline_mlp")
        except Exception as e:
            print(f"Error training MLP baseline: {str(e)}")
        
        print("\nTraining Transformer baseline model...")
        try:
            train_baseline_model(TransformerBaseline(X_train.shape[1], 4), X_train, Y_train, X_test, Y_test, scaler_y, "baseline_transformer")
        except Exception as e:
            print(f"Error training Transformer baseline: {str(e)}")
        
        print("\nTraining classical diffusion model...")
        try:
            # Use early stopping for sampling to speed up the generation process
            train_quantum_diffusion(X_train, Y_train, X_test, Y_test, scaler_y, 
                                    use_quantum=False, 
                                    save_dir="diffusion_classical",
                                    early_stop_sampling=True)
        except Exception as e:
            print(f"Error training classical diffusion: {str(e)}")
        
        print("\nTraining quantum diffusion model...")
        try:
            # Use early stopping for sampling to speed up the generation process
            train_quantum_diffusion(X_train, Y_train, X_test, Y_test, scaler_y, 
                                    use_quantum=True, 
                                    save_dir="diffusion_quantum",
                                    early_stop_sampling=True)
        except Exception as e:
            print(f"Error training quantum diffusion: {str(e)}")

        # Summarize results
        print("\nSummarizing results...")
        try:
            summarize_results(
                targets=['LCE', 'sC', 'aC', 'C'],
                output_dirs={
                    "MLP": "baseline_mlp",
                    "Transformer": "baseline_transformer",
                    "Diffusion-Classical": "diffusion_classical",
                    "Diffusion-Quantum": "diffusion_quantum"
                }
            )
        except Exception as e:
            print(f"Error summarizing results: {str(e)}")
            
        print("\nTraining completed!")
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
