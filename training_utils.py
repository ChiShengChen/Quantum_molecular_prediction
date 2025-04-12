import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd # Needed for run_sklearn_pipeline's eval call indirectly

# Imports for run_sklearn_pipeline (handle potential import errors)
try:
    from sklearn.multioutput import MultiOutputRegressor
    sklearn_available = True
except ImportError:
    MultiOutputRegressor = None
    sklearn_available = False

try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    xgb = None
    xgboost_available = False

try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lgb = None
    lightgbm_available = False
    
# Import evaluation function (assuming it will be in evaluation_utils.py)
# Need to handle potential circular dependency if evaluation_utils also imports this
# A better structure might put evaluate_predictions in its own file or a common utils file.
# For now, assume it's importable.
try:
     from evaluation_utils import evaluate_predictions
except ImportError:
     print("Warning: Could not import evaluate_predictions from evaluation_utils. Make sure evaluation_utils.py exists.")
     # Define a dummy function if import fails to avoid crashing run_sklearn_pipeline
     def evaluate_predictions(*args, **kwargs):
          print("Error: evaluate_predictions function not found!")
          return None

print("Initializing Training Utils Module...")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Generic Training Function for DL Models ---
def train_model(
    model, model_name, # Added model_name for saving
    X_train, Y_train, X_val, Y_val,
    epochs=50, batch_size=32, lr=1e-3,
    save_dir_prefix="results", # Use prefix for save dir
    early_stop_patience=5,
    weight_decay=1e-4 # Added weight_decay parameter
):
    """
    Generic training function for PyTorch DL models.
    Uses AdamW optimizer with specified weight decay.
    """
    save_dir = f"{save_dir_prefix}_{model_name}" # Create specific save dir
    os.makedirs(save_dir, exist_ok=True)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Ensure data are tensors on the correct device
    X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
    X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    print(f"Starting {model_name} training for {epochs} epochs (lr={lr}, wd={weight_decay}, saving to {save_dir})...") # Log lr and wd
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{model_name} Train]", leave=False) as tepoch:
            for i, (x_batch, y_batch) in enumerate(tepoch):
                # Data already on DEVICE
                if torch.isnan(x_batch).any() or torch.isnan(y_batch).any():
                    continue

                optimizer.zero_grad()
                predictions = model(x_batch)
                
                if torch.isnan(predictions).any():
                     print(f"Warning: NaN prediction in {model_name} epoch {epoch+1}, batch {i}. Skipping.")
                     continue

                loss = loss_fn(predictions, y_batch)
                
                if torch.isnan(loss):
                     print(f"Warning: NaN loss in {model_name} epoch {epoch+1}, batch {i}. Skipping.")
                     continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    # Data already on DEVICE
                    val_preds = model(x_val_batch)
                    val_loss = loss_fn(val_preds, y_val_batch)
                    if not torch.isnan(val_loss):
                         epoch_val_losses.append(val_loss.item())

        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses) if epoch_val_losses else float('inf')
        val_losses.append(avg_val_loss)
        
        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} [{model_name}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}") # Log LR

        # Save loss curves
        try:
             plt.figure(figsize=(10, 5))
             plt.plot(train_losses, label='Train Loss')
             plt.plot(val_losses, label='Validation Loss')
             plt.xlabel('Epoch')
             plt.ylabel('MSE Loss')
             plt.title(f'{model_name} Training Loss')
             plt.legend()
             plt.grid(True)
             plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
             plt.close()
        except Exception as e:
             print(f"Warning: Could not save loss curve plot for {model_name}: {e}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            try:
                 torch.save({
                     'model_state_dict': best_model_state,
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state
                     'epoch': epoch,
                     'loss': best_val_loss,
                 }, os.path.join(save_dir, 'best_model.pt'))
                 print(f"Epoch {epoch+1} [{model_name}] - Best model saved with validation loss: {best_val_loss:.6f}")
            except Exception as e:
                 print(f"Warning: Could not save best model checkpoint for {model_name}: {e}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered for {model_name} after {epoch+1} epochs.")
                if best_model_state:
                    try:
                         model.load_state_dict(best_model_state)
                    except Exception as e:
                         print(f"Warning: Could not load best model state for {model_name} after early stopping: {e}")
                else:
                    print(f"Warning: Early stopping triggered for {model_name} but no best model state was saved.")
                break

    # Save final model
    try:
         torch.save({
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict(),
             'epoch': epoch,
             'loss': avg_val_loss,
         }, os.path.join(save_dir, 'final_model.pt'))
    except Exception as e:
         print(f"Warning: Could not save final model checkpoint for {model_name}: {e}")

    print(f"{model_name} training finished.")
    # Load best model state for final return if it exists
    if best_model_state:
         try:
             model.load_state_dict(best_model_state)
             print(f"Loaded best model state for {model_name} for final evaluation.")
         except Exception as e:
              print(f"Warning: Could not load final best model state for {model_name}: {e}")
              
    print(f"[DEBUG] Returning type from train_model: {type(model)}") # Add type check
    return model # Only return the trained model object

# --- Sklearn/XGBoost/LightGBM Pipeline ---
def run_sklearn_pipeline(base_model_class, model_name, model_args, run_flag,
                         X_train_np, Y_train_np, X_test_np, Y_test_np,
                         scaler_y, target_names, save_dir_prefix):
    """
    Runs the training and evaluation pipeline for sklearn-like models.
    Handles multi-output regression using MultiOutputRegressor.
    """
    if not run_flag:
        print(f"\n--- Skipping {model_name} Model --- ")
        return None

    # Check if necessary libraries are available
    lib_available = True
    if model_name in ["SVR", "KNN"] and not sklearn_available:
        lib_available = False
        print(f"\n--- Skipping {model_name} Model (scikit-learn not available) --- ")
    elif model_name == "XGBoost" and not xgboost_available:
        lib_available = False
        print(f"\n--- Skipping {model_name} Model (xgboost not available) --- ")
    elif model_name == "LightGBM" and not lightgbm_available:
        lib_available = False
        print(f"\n--- Skipping {model_name} Model (lightgbm not available) --- ")
    elif not MultiOutputRegressor:
        lib_available = False
        print(f"\n--- Skipping {model_name} Model (MultiOutputRegressor from sklearn not available) --- ")
        
    if not lib_available:
         return None

    print(f"\n--- {model_name} Model --- ")
    save_dir = f"{save_dir_prefix}_{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Initialize the base model
        base_model = base_model_class(**model_args)
        # Wrap with MultiOutputRegressor for multi-target prediction
        # Use n_jobs=-1 to utilize all available CPU cores for fitting parallelizable base estimators
        model = MultiOutputRegressor(base_model, n_jobs=-1) 
        print(f"Initialized {model_name} model (wrapped with MultiOutputRegressor):")
        print(model)

        # Train the model
        print(f"Training {model_name} model...")
        # Make sure Y_train_np is 2D
        if Y_train_np.ndim == 1:
             Y_train_np = Y_train_np.reshape(-1, 1)
        model.fit(X_train_np, Y_train_np)
        print(f"{model_name} training finished.")

        # Generate predictions
        print(f"Generating {model_name} predictions...")
        preds_scaled = model.predict(X_test_np)

        # Inverse transform predictions
        try:
            # Ensure preds_scaled is 2D before inverse_transform
            if preds_scaled.ndim == 1:
                 preds_scaled = preds_scaled.reshape(-1, 1)
            preds_denorm = scaler_y.inverse_transform(preds_scaled)
        except Exception as e:
            print(f"Warning: {model_name} inverse transform failed: {e}. Using scaled predictions.")
            preds_denorm = preds_scaled

        # Evaluate predictions
        print(f"Evaluating {model_name} predictions...")
        metrics = evaluate_predictions(
            predictions_denorm=preds_denorm, Y_test_denorm=Y_test_np,
            target_names=target_names, model_name=model_name, save_dir=save_dir
        )
        return metrics

    except Exception as e:
        print(f"\n--- Error during {model_name} pipeline: {e} --- ")
        import traceback
        traceback.print_exc()
        return None 