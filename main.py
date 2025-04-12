import os
import torch
import numpy as np
import warnings
import argparse
import traceback
import torch.nn.init as init # Import init
import statistics # For mean/stdev calculation

# --- Import necessary modules ---
print("Importing modules...")
from data_utils import load_molecular_data
from quantum_layers import EnhancedQuantumLayer # Keep if QuantumRegressionNet uses it directly
from quantum_models import QuantumRegressionNet, DampedStyleRegressionModel, QLSTMBaseline
from classical_models import MLPBaseline, TransformerBaseline, CNNBaseline, ResNetBaseline, LSTMBaseline
from training_utils import train_model, run_sklearn_pipeline
from evaluation_utils import evaluate_predictions, summarize_results

# Imports for sklearn/boosting models (needed for run_sklearn_pipeline call)
try:
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    sklearn_available = True
except ImportError:
    SVR, KNeighborsRegressor = None, None
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
# --- End Imports ---

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Helper function to aggregate metrics (can be placed inside main or imported)
def aggregate_run_metrics(run_metrics_list, target_names, metric_keys=['RMSE', 'MAE', 'R2']):
    """ Aggregates metrics from multiple runs into mean and std dev. """
    if not run_metrics_list:
        return None

    num_runs = len(run_metrics_list)
    aggregated = {'mean': {t: {} for t in target_names}, 'std': {t: {} for t in target_names}}

    for target in target_names:
        for metric_key in metric_keys:
            values = []
            for run_result in run_metrics_list:
                # Check if target and metric exist for the run
                if run_result is not None and target in run_result and metric_key in run_result[target]:
                     value = run_result[target][metric_key]
                     if value is not None and not np.isnan(value):
                          values.append(value)
                # else: handle missing data if necessary, maybe append NaN or skip?
                # For now, only aggregate valid, non-NaN values.

            if len(values) > 1: # Need at least 2 points for stdev
                try:
                    aggregated['mean'][target][metric_key] = statistics.mean(values)
                    aggregated['std'][target][metric_key] = statistics.stdev(values)
                except statistics.StatisticsError:
                    print(f"StatisticsError calculating mean/std for {target}/{metric_key}. Setting to NaN.")
                    aggregated['mean'][target][metric_key] = np.nan
                    aggregated['std'][target][metric_key] = np.nan
            elif len(values) == 1:
                aggregated['mean'][target][metric_key] = values[0]
                aggregated['std'][target][metric_key] = 0 # No variance with one point
                print(f"Warning: Only 1 valid run found for {target}/{metric_key}. Std dev set to 0.")
            else: # No valid data points
                aggregated['mean'][target][metric_key] = np.nan
                aggregated['std'][target][metric_key] = np.nan
                print(f"Warning: No valid runs found for {target}/{metric_key}. Mean/Std dev set to NaN.")

    return aggregated


# --- Main Execution Logic ---
def main():
    # Note: QASA global params are now handled within the respective modules or passed explicitly.
    print("=" * 80)
    print("Quantum & Classical Regression Model Training")
    print("=" * 80)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Quantum and Classical Regression Models")
    # Add all arguments from the previous version...
    # Common args
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (faster training, smaller models, 1 run)") # Debug implies 1 run
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs with different seeds for DL models") # Added num_runs
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed for DL model runs") # Added base_seed
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (for DL models)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (for DL models)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (for DL models)")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate for DL models")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
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
    # QASA (Damped Style) specific args
    parser.add_argument("--qasa-hidden-dim", type=int, default=128, help="Hidden dimension for QASA-style Transformer layers")
    parser.add_argument("--qasa-nhead", type=int, default=4, help="Number of heads for QASA-style attention layers")
    parser.add_argument("--qasa-num-layers", type=int, default=4, help="Total number of encoder layers in QASA-style model (N-1 classical + 1 quantum)")
    parser.add_argument("--qasa-n-qubits", type=int, default=8, help="Number of primary qubits for QASA-style quantum circuit (uses n_qubits+1 wires)")
    parser.add_argument("--qasa-circuit-layers", type=int, default=4, help="Number of layers for QASA-style quantum circuit")
    # Baseline specific args (sklearn/Boosting)
    parser.add_argument("--svr-c", type=float, default=1.0, help="Regularization parameter C for SVR")
    parser.add_argument("--svr-kernel", type=str, default='rbf', help="Kernel type for SVR ('linear', 'rbf', etc.)")
    parser.add_argument("--knn-neighbors", type=int, default=5, help="Number of neighbors for KNN")
    parser.add_argument("--xgb-n-estimators", type=int, default=100, help="Number of boosting rounds for XGBoost")
    parser.add_argument("--xgb-lr", type=float, default=0.1, help="Learning rate for XGBoost")
    parser.add_argument("--lgbm-n-estimators", type=int, default=100, help="Number of boosting rounds for LightGBM")
    parser.add_argument("--lgbm-lr", type=float, default=0.1, help="Learning rate for LightGBM")
    # QLSTM specific args
    parser.add_argument("--qlstm-hidden-dim", type=int, default=128, help="Hidden dimension size for QLSTM")
    parser.add_argument("--qlstm-n-qubits", type=int, default=4, help="Number of qubits in QuantumLSTMCell PQC")
    parser.add_argument("--qlstm-circuit-layers", type=int, default=1, help="Number of layers in QuantumLSTMCell PQC")

    # Flags to run specific models
    parser.add_argument("--run-qnn", action=argparse.BooleanOptionalAction, default=True, help="Run the original QNN model")
    parser.add_argument("--run-mlp", action=argparse.BooleanOptionalAction, default=True, help="Run the MLP baseline model")
    parser.add_argument("--run-transformer", action=argparse.BooleanOptionalAction, default=True, help="Run the Transformer baseline model")
    parser.add_argument("--run-cnn", action=argparse.BooleanOptionalAction, default=True, help="Run the CNN baseline model")
    parser.add_argument("--run-resnet", action=argparse.BooleanOptionalAction, default=True, help="Run the ResNet baseline model")
    parser.add_argument("--run-lstm", action=argparse.BooleanOptionalAction, default=True, help="Run the LSTM baseline model")
    parser.add_argument("--run-qasa", action=argparse.BooleanOptionalAction, default=True, help="Run the QASA (Damped Style) model")
    parser.add_argument("--run-svr", action=argparse.BooleanOptionalAction, default=True, help="Run the SVR baseline model")
    parser.add_argument("--run-knn", action=argparse.BooleanOptionalAction, default=True, help="Run the KNN baseline model")
    parser.add_argument("--run-xgboost", action=argparse.BooleanOptionalAction, default=True, help="Run the XGBoost baseline model")
    parser.add_argument("--run-lightgbm", action=argparse.BooleanOptionalAction, default=True, help="Run the LightGBM baseline model")
    parser.add_argument("--run-qlstm", action=argparse.BooleanOptionalAction, default=True, help="Run the Quantum LSTM baseline model")

    args = parser.parse_args()
    base_seed = args.base_seed # Store base seed
    num_runs = 1 if args.debug else args.num_runs # Force 1 run in debug mode
    if args.debug and args.num_runs != 1:
        print("Warning: Debug mode enabled, overriding --num-runs to 1.")

    # --- Parameter Setup (Debug vs Normal) ---
    if args.debug:
        print("\nRunning in DEBUG mode - reduced training parameters")
        epochs = 5
        batch_size = 16
        patience = 2
        lr = 5e-5 # Default LR for debug
        dropout_rate = 0.1 # Default dropout for debug
        weight_decay = 1e-4 # Default weight decay for debug
        # QNN
        qnn_hidden_dim = 128; n_qubits = 4; n_layers = 2
        # DL Baselines
        mlp_hidden_dim = 128
        transformer_hidden_dim = 64; transformer_nhead= 2; transformer_nlayers = 1
        cnn_channels = [16, 32]; cnn_kernel_size = 3
        resnet_hidden_dim = 64; resnet_blocks = 2
        lstm_hidden_dim = 64; lstm_layers = 1
        # QASA (Damped Style)
        qasa_hidden_dim = 64; qasa_nhead = 2; qasa_num_layers = 2; qasa_n_qubits = 4; qasa_circuit_layers = 2
        # Sklearn/Boosting Baselines
        svr_c = 0.5; svr_kernel = args.svr_kernel
        knn_neighbors = 3
        xgb_n_estimators = 20; xgb_lr = args.xgb_lr
        lgbm_n_estimators = 20; lgbm_lr = args.lgbm_lr
        # QLSTM
        qlstm_hidden_dim = 64; qlstm_n_qubits = 4; qlstm_circuit_layers = 1
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        patience = 5
        lr = args.lr
        dropout_rate = args.dropout_rate
        weight_decay = args.weight_decay
        # QNN
        qnn_hidden_dim = args.qnn_hidden_dim; n_qubits = args.n_qubits; n_layers = args.n_layers
        # DL Baselines
        mlp_hidden_dim = args.mlp_hidden_dim
        transformer_hidden_dim = args.transformer_hidden_dim; transformer_nhead = args.transformer_nhead; transformer_nlayers = args.transformer_nlayers
        cnn_channels = args.cnn_channels; cnn_kernel_size = args.cnn_kernel_size
        resnet_hidden_dim = args.resnet_hidden_dim; resnet_blocks = args.resnet_blocks
        lstm_hidden_dim = args.lstm_hidden_dim; lstm_layers = args.lstm_layers
        # QASA (Damped Style)
        qasa_hidden_dim = args.qasa_hidden_dim; qasa_nhead = args.qasa_nhead; qasa_num_layers = args.qasa_num_layers; qasa_n_qubits = args.qasa_n_qubits; qasa_circuit_layers = args.qasa_circuit_layers
        # Sklearn/Boosting Baselines
        svr_c = args.svr_c; svr_kernel = args.svr_kernel
        knn_neighbors = args.knn_neighbors
        xgb_n_estimators = args.xgb_n_estimators; xgb_lr = args.xgb_lr
        lgbm_n_estimators = args.lgbm_n_estimators; lgbm_lr = args.lgbm_lr
        # QLSTM
        qlstm_hidden_dim = args.qlstm_hidden_dim; qlstm_n_qubits = args.qlstm_n_qubits; qlstm_circuit_layers = args.qlstm_circuit_layers

    # --- Data Loading ---
    try:
        target_columns = ['LCE', 'sC', 'aC', 'C']
        print("\nLoading molecular data...")
        (
            X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_numpy,
            scaler_y, feature_names, target_names
        ) = load_molecular_data(args.excel_path, args.sheet_name, target_columns)

        input_dim = X_train_tensor.shape[1]
        output_dim = Y_train_tensor.shape[1]

        # Get numpy versions for sklearn models
        X_train_np = X_train_tensor.numpy()
        Y_train_np = Y_train_tensor.numpy() # Scaled Y_train
        X_test_np = X_test_tensor.numpy()
        # Y_test_numpy is already the original unscaled Y_test

        # Create validation set (tensors) for DL models
        if len(X_train_tensor) > 10:
             val_size = min(max(1, int(0.1 * len(X_train_tensor))), 500)
             indices = torch.randperm(len(X_train_tensor)).tolist()
             X_val_tensor, Y_val_tensor = X_train_tensor[indices[:val_size]], Y_train_tensor[indices[:val_size]]
             X_train_tensor, Y_train_tensor = X_train_tensor[indices[val_size:]], Y_train_tensor[indices[val_size:]]
             print(f"\nUsing {len(X_val_tensor)} samples for validation (DL models).")
        else:
             X_val_tensor, Y_val_tensor = X_train_tensor, Y_train_tensor
             print("\nWarning: Small dataset, using all training data for validation (DL models).")

        print(f"Training set (DL): {X_train_tensor.shape[0]} samples")
        print(f"Validation set (DL): {X_val_tensor.shape[0]} samples")
        print(f"Test set: {X_test_tensor.shape[0]} samples")
        print(f"Input dimensions: {input_dim}")
        print(f"Output dimensions: {output_dim}")

        all_metrics = {} # Store metrics from all models

        # --- Prepare DL Model Configurations --- (Moved population outside loop)
        dl_models_to_run_configs = []
        # QNN
        if args.run_qnn:
            dl_models_to_run_configs.append({
                'model_class': QuantumRegressionNet,
                'model_name': "QNN",
                'model_args': {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': qnn_hidden_dim, 'n_qubits': n_qubits, 'n_layers': n_layers, 'dropout_rate': dropout_rate}
            })
        # MLP
        if args.run_mlp:
            dl_models_to_run_configs.append({
                'model_class': MLPBaseline,
                'model_name': "MLP",
                'model_args': {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': mlp_hidden_dim, 'dropout_rate': dropout_rate}
            })
        # Transformer
        if args.run_transformer:
            current_transformer_hidden_dim = transformer_hidden_dim
            if current_transformer_hidden_dim % transformer_nhead != 0:
                original_dim = current_transformer_hidden_dim
                current_transformer_hidden_dim = (current_transformer_hidden_dim // transformer_nhead) * transformer_nhead
                if current_transformer_hidden_dim == 0: current_transformer_hidden_dim = transformer_nhead
                print(f"Adjusted Transformer hidden_dim from {original_dim} to {current_transformer_hidden_dim}...")
            dl_models_to_run_configs.append({
                'model_class': TransformerBaseline,
                'model_name': "Transformer",
                'model_args': {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': current_transformer_hidden_dim, 'nhead': transformer_nhead, 'num_layers': transformer_nlayers, 'dropout_rate': dropout_rate}
            })
        # CNN
        if args.run_cnn:
            dl_models_to_run_configs.append({
                'model_class': CNNBaseline,
                'model_name': "CNN",
                'model_args': {'input_dim': input_dim, 'output_dim': output_dim, 'num_channels': cnn_channels, 'kernel_size': cnn_kernel_size, 'dropout_rate': dropout_rate}
            })
        # ResNet
        if args.run_resnet:
            dl_models_to_run_configs.append({
                'model_class': ResNetBaseline,
                'model_name': "ResNet",
                'model_args': {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': resnet_hidden_dim, 'num_blocks': resnet_blocks, 'dropout_rate': dropout_rate}
            })
        # LSTM
        if args.run_lstm:
            dl_models_to_run_configs.append({
                'model_class': LSTMBaseline,
                'model_name': "LSTM",
                'model_args': {'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dim': lstm_hidden_dim, 'num_layers': lstm_layers, 'dropout_rate': dropout_rate}
            })
        # QASA (Damped Style)
        if args.run_qasa:
            # Ensure num_layers >= 2 for this structure (N-1 classical + 1 quantum)
            if qasa_num_layers < 2:
                 print(f"Warning: QASA requires at least 2 layers (num_layers = {qasa_num_layers}). Setting to 2.")
                 qasa_num_layers = 2
            dl_models_to_run_configs.append({
                'model_class': DampedStyleRegressionModel,
                'model_name': "QASA",
                'model_args': {'input_dim': input_dim, 'output_dim': output_dim,
                               'hidden_dim': qasa_hidden_dim, 'num_layers': qasa_num_layers,
                               'nhead': qasa_nhead,
                               'n_qubits': qasa_n_qubits, 'circuit_layers': qasa_circuit_layers,
                               'dropout_rate': dropout_rate}
            })
        # QLSTM
        if args.run_qlstm:
            dl_models_to_run_configs.append({
                'model_class': QLSTMBaseline,
                'model_name': "QLSTM",
                'model_args': {'input_dim': input_dim, 'output_dim': output_dim,
                               'hidden_dim': qlstm_hidden_dim,
                               'n_qubits': qlstm_n_qubits,
                               'qlstm_layers': qlstm_circuit_layers,
                               'num_lstm_layers': 1, # Fixed to 1 layer
                               'dropout_rate': dropout_rate}
            })

        # --- Run DL Models (Loop through configured models) ---
        for model_config in dl_models_to_run_configs:
            model_class = model_config['model_class']
            model_name = model_config['model_name']
            model_args = model_config['model_args']
            model_runs_metrics = [] # Store metrics for each run of *this* model

            print(f"\n--- Running {model_name} Model ({num_runs} runs, Base Seed: {base_seed}) --- ")

            # Loop for multiple runs with different seeds
            for run_idx in range(num_runs):
                current_seed = base_seed + run_idx
                print(f"\n--- {model_name} Run {run_idx + 1}/{num_runs} (Seed: {current_seed}) --- ")
                # Set seeds for reproducibility
                torch.manual_seed(current_seed)
                np.random.seed(current_seed)
                # Consider adding torch.cuda.manual_seed_all(current_seed) if using CUDA

                save_dir = f"{args.save_dir_prefix}_{model_name}_run{run_idx+1}" # Unique dir per run
                os.makedirs(save_dir, exist_ok=True)

                try:
                    model = model_class(**model_args)

                    # Train the model
                    trained_model = train_model(
                        model=model,
                        model_name=f"{model_name}_run{run_idx+1}", # Unique name for logs/saves per run
                        X_train=X_train_tensor,
                        Y_train=Y_train_tensor,
                        X_val=X_val_tensor,
                        Y_val=Y_val_tensor,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        save_dir_prefix=args.save_dir_prefix, # Base prefix for saving structure
                        early_stop_patience=patience,
                        weight_decay=weight_decay
                    )

                    # Evaluation
                    trained_model.eval()
                    with torch.no_grad():
                        predictions_scaled = trained_model(X_test_tensor.to(DEVICE))
                    predictions_scaled_np = predictions_scaled.cpu().numpy()
                    predictions_denorm = scaler_y.inverse_transform(predictions_scaled_np)

                    # Evaluate and store metrics for this run
                    metrics = evaluate_predictions(predictions_denorm, Y_test_numpy, target_names,
                                                   f"{model_name}_run{run_idx+1}", save_dir)
                    if metrics:
                        model_runs_metrics.append(metrics)
                        print(f"Finished evaluation for {model_name} Run {run_idx + 1}.")
                    else:
                        print(f"Warning: Evaluation failed or produced no metrics for {model_name} Run {run_idx + 1}.")
                        model_runs_metrics.append(None) # Append None if metrics failed

                except Exception as e:
                     print(f"Error running {model_name} Run {run_idx + 1}: {e}")
                     traceback.print_exc()
                     model_runs_metrics.append(None) # Append None if run failed

            # --- Aggregate metrics across runs for this model ---
            if model_runs_metrics:
                # Filter out None values before aggregation
                successful_run_metrics = [m for m in model_runs_metrics if m is not None]
                if successful_run_metrics:
                    aggregated_metrics = aggregate_run_metrics(successful_run_metrics, target_names)
                    if aggregated_metrics:
                         all_metrics[model_name] = aggregated_metrics # Store {'mean': {...}, 'std': {...}}
                         print(f"\nAggregated metrics for {model_name} across {len(successful_run_metrics)} successful runs.")
                    else:
                         print(f"Warning: Failed to aggregate metrics for {model_name}.")
                else:
                     print(f"No successful runs with metrics to aggregate for {model_name}.")
            else:
                print(f"\nNo runs attempted or all failed for {model_name}.")

        # --- Run Sklearn/Boosting Model Pipelines (Run once) ---
        # Store results directly without 'mean'/'std' structure
        if args.run_svr:
            svr_metrics = run_sklearn_pipeline(
                SVR, "SVR",
                {'C': svr_c, 'kernel': svr_kernel}, args.run_svr,
                X_train_np, Y_train_np, X_test_np, Y_test_numpy,
                scaler_y, target_names, args.save_dir_prefix
            )
            if svr_metrics: all_metrics["SVR"] = svr_metrics
        if args.run_knn:
            knn_metrics = run_sklearn_pipeline(
                KNeighborsRegressor, "KNN",
                {'n_neighbors': knn_neighbors}, args.run_knn,
                X_train_np, Y_train_np, X_test_np, Y_test_numpy,
                scaler_y, target_names, args.save_dir_prefix
            )
            if knn_metrics: all_metrics["KNN"] = knn_metrics
        if args.run_xgboost:
            xgb_metrics = run_sklearn_pipeline(
                xgb.XGBRegressor, "XGBoost",
                {'n_estimators': xgb_n_estimators, 'learning_rate': xgb_lr, 'objective': 'reg:squarederror', 'random_state': base_seed}, # Use base_seed for sklearn models too
                args.run_xgboost,
                X_train_np, Y_train_np, X_test_np, Y_test_numpy,
                scaler_y, target_names, args.save_dir_prefix
            )
            if xgb_metrics: all_metrics["XGBoost"] = xgb_metrics
        if args.run_lightgbm:
            lgbm_metrics = run_sklearn_pipeline(
                lgb.LGBMRegressor, "LightGBM",
                {'n_estimators': lgbm_n_estimators, 'learning_rate': lgbm_lr, 'random_state': base_seed}, # Use base_seed
                args.run_lightgbm,
                X_train_np, Y_train_np, X_test_np, Y_test_numpy,
                scaler_y, target_names, args.save_dir_prefix
            )
            if lgbm_metrics: all_metrics["LightGBM"] = lgbm_metrics

        # --- Summarize Results ---
        if len(all_metrics) > 0: # Check if any metrics were collected
            summarize_results(all_metrics, target_names, args.save_dir_prefix)
        else:
             print("\nNo models produced metrics, skipping summary.")

    except FileNotFoundError:
         print(f"Error: Excel file not found at {args.excel_path}")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 