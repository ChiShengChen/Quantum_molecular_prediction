import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statistics # Needed if aggregate_run_metrics is moved here

print("Initializing Evaluation Utils Module...")

# --- Evaluation Function ---
def evaluate_predictions(predictions_denorm, Y_test_denorm, target_names, model_name, save_dir):
    """
    Evaluate predictions and create visualizations.
    Assumes Y_test_denorm is the original (unscaled) test set targets.
    Assumes predictions_denorm are the model's predictions, already inverse-transformed.
    """
    os.makedirs(save_dir, exist_ok=True) # save_dir is the full path
    metrics = {target: {} for target in target_names}

    # Ensure predictions have the same number of columns as targets
    if predictions_denorm.shape[1] != len(target_names):
         print(f"Error: Number of prediction columns ({predictions_denorm.shape[1]}) does not match number of targets ({len(target_names)}) for model {model_name}.")
         # Return NaN metrics for all targets
         for target in target_names:
              metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
         return metrics

    for i, target in enumerate(target_names):
        y_true = Y_test_denorm[:, i]
        y_pred = predictions_denorm[:, i]

        # Remove NaN/Inf values from both true and predicted values
        valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        
        if not np.any(valid_idx):
            print(f"Warning: No valid (non-NaN/Inf) predictions/targets found for {target} in {model_name}")
            metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
            continue

        y_true_valid = y_true[valid_idx]
        y_pred_valid = y_pred[valid_idx]

        # Check if enough valid points remain for metric calculation
        if len(y_true_valid) < 2: # Need at least 2 points for R2 score
             print(f"Warning: Not enough valid points ({len(y_true_valid)}) for {target} metrics in {model_name}.")
             metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
             continue

        # Calculate metrics
        try:
             mse = mean_squared_error(y_true_valid, y_pred_valid)
             rmse = np.sqrt(mse)
             mae = mean_absolute_error(y_true_valid, y_pred_valid)
             # Handle potential case where variance of y_true_valid is zero
             if np.var(y_true_valid) < 1e-9:
                  r2 = np.nan # R2 is ill-defined if true values have no variance
                  print(f"Warning: R2 score is NaN for target '{target}' in model {model_name} because true values have near-zero variance.")
             else:
                  r2 = r2_score(y_true_valid, y_pred_valid)

             metrics[target] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

             # Create scatter plot
             plt.figure(figsize=(8, 8))
             plt.scatter(y_true_valid, y_pred_valid, alpha=0.5, label=f'{model_name}')
             # Add identity line based on valid data range
             min_val = min(y_true_valid.min(), y_pred_valid.min())
             max_val = max(y_true_valid.max(), y_pred_valid.max())
             plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
             plt.xlabel(f"True {target}")
             plt.ylabel(f"Predicted {target}")
             # Handle NaN R2 in title
             r2_title = f"{r2:.4f}" if not np.isnan(r2) else "NaN"
             plt.title(f"{target} - {model_name} (R² = {r2_title})")
             plt.legend()
             plt.grid(True)
             plt.savefig(os.path.join(save_dir, f"scatter_{target}.png"))
             plt.close()
             
        except Exception as e:
            print(f"Error calculating/plotting metrics for {target} in {model_name}: {e}")
            metrics[target] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

    # Save metrics to CSV
    try:
         metrics_df = pd.DataFrame(metrics).T # Transpose for better readability
         metrics_df.index.name = 'Target'
         metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"))
         print(f"--- {model_name} Metrics --- ")
         print(metrics_df)
    except Exception as e:
         print(f"Warning: Could not save metrics.csv for {model_name}: {e}")

    # Save predictions (already denormalized)
    try:
         pd.DataFrame(predictions_denorm, columns=target_names).to_csv(
             os.path.join(save_dir, "predictions.csv"), index=False
         )
         print(f"Predictions for {model_name} saved to {os.path.join(save_dir, 'predictions.csv')}")
    except Exception as e:
         print(f"Warning: Could not save predictions.csv for {model_name}: {e}")

    return metrics

# --- Result Summarization (Modified for CSV & Target Plots) ---
def summarize_results(all_metrics, target_names, save_dir_prefix="results"):
    """
    Generates comparison plots and saves summary metrics to CSV.
    Includes overall comparison and target-specific plots.
    Handles aggregated (mean/std) and single-run metrics.
    """
    print("\n--- Results Summary --- ")
    valid_metrics = {name: metrics for name, metrics in all_metrics.items() if metrics is not None}
    if not valid_metrics:
         print("No valid model metrics found to summarize.")
         return

    # --- Define Model Orders ---
    full_model_order = [
        'QASA', 'QLSTM', 'QNN', 'Transformer', 'ResNet', 'MLP',
        'SVR', 'KNN', 'LightGBM', 'XGBoost'
    ]
    subset_model_order = [
        'QASA', 'QLSTM', 'QNN', 'Transformer', 'ResNet', 'MLP',
        'SVR', 'KNN'
    ]

    metric_names = ['RMSE', 'MAE', 'R2']
    summary_plot_dir = f"{save_dir_prefix}_summary"
    os.makedirs(summary_plot_dir, exist_ok=True)

    # --- Helper function to generate plots (Overall Comparison) ---
    def generate_summary_plot(plot_model_names, plot_data_mean, plot_data_std, plot_title_suffix, plot_filename_prefix):
        if not plot_model_names:
            print(f"No models found for {plot_title_suffix} plot.")
            return

        custom_colors = ['#005CAF', '#FC9F4D', '#A8D8B9', '#F4A7B9']

        for metric in metric_names:
            plt.figure(figsize=(max(10, len(plot_model_names) * 1.5), 7))
            num_models = len(plot_model_names)
            bar_width = 0.8 / len(target_names) if len(target_names) > 0 else 0.8
            index = np.arange(num_models)

            all_nan_for_metric = True
            for i, target in enumerate(target_names):
                if target not in plot_data_mean[metric]:
                    print(f"Warning: Target '{target}' not found in summary data for metric '{metric}'. Skipping.")
                    continue
                mean_values = plot_data_mean[metric][target]
                std_values = plot_data_std[metric][target]
                if len(mean_values) != num_models or len(std_values) != num_models:
                     print(f"Warning: Length mismatch for metric '{metric}', target '{target}'. Skipping bar.")
                     continue
                plot_heights = [val if not pd.isna(val) else 0 for val in mean_values]
                plot_errors = [val if not pd.isna(val) else 0 for val in std_values]
                has_valid_data = any(not pd.isna(val) for val in mean_values)
                if has_valid_data:
                    all_nan_for_metric = False
                    color_index = i % len(custom_colors)
                    plt.bar(index + i * bar_width, plot_heights, bar_width, label=target,
                            color=custom_colors[color_index], yerr=plot_errors, capsize=4)

            if not all_nan_for_metric:
                plt.xlabel('Model', fontweight='bold')
                plt.ylabel(metric)
                title = f'{metric} Comparison {plot_title_suffix} (Error bars = ±1 std dev)'
                plt.title(title, fontweight='bold', fontsize=14)
                plt.xticks(index + bar_width * (len(target_names) - 1) / 2, plot_model_names, rotation=15, ha="right")
                if target_names:
                    plt.legend(title='Targets', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout(rect=[0, 0, 0.9, 1])
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                summary_plot_path = os.path.join(summary_plot_dir, f"{plot_filename_prefix}{metric}.png")
                try:
                    plt.savefig(summary_plot_path)
                    print(f"Saved summary plot: {summary_plot_path}")
                except Exception as e:
                    print(f"Warning: Could not save summary plot {summary_plot_path}: {e}")
            else:
                print(f"Skipping summary plot for {metric} ({plot_title_suffix}) as no valid data was found.")
            plt.close()

    # --- Helper function to generate Target-Specific plots --- 
    def generate_target_specific_plot(target_name, plot_model_names, plot_data_mean, plot_data_std):
        if not plot_model_names:
            print(f"No models found for target-specific plot: {target_name}.")
            return
            
        # Use a different color scheme or fixed colors for metrics
        metric_colors = {'RMSE': '#005CAF', 'MAE': '#FC9F4D', 'R2': '#A8D8B9'} 

        plt.figure(figsize=(max(10, len(plot_model_names) * 1.2), 7)) # Adjust figsize if needed
        num_models = len(plot_model_names)
        bar_width = 0.8 / len(metric_names) # Width based on number of metrics
        index = np.arange(num_models)
        all_nan_for_target = True

        for i, metric in enumerate(metric_names):
            # Get mean and std dev values for this metric across ordered models for the specific target
            mean_values = plot_data_mean[metric][target_name]
            std_values = plot_data_std[metric][target_name]
            
            if len(mean_values) != num_models or len(std_values) != num_models:
                print(f"Warning: Length mismatch for metric '{metric}', target '{target_name}'. Skipping bar.")
                continue
                
            plot_heights = [val if not pd.isna(val) else 0 for val in mean_values]
            plot_errors = [val if not pd.isna(val) else 0 for val in std_values]
            has_valid_data = any(not pd.isna(val) for val in mean_values)

            if has_valid_data:
                all_nan_for_target = False
                color = metric_colors.get(metric, 'gray') # Default color if metric not in map
                plt.bar(index + i * bar_width, plot_heights, bar_width, label=metric,
                        color=color, yerr=plot_errors, capsize=4)

        if not all_nan_for_target:
            plt.xlabel('Model', fontweight='bold')
            plt.ylabel('Metric Value')
            title = f'Model Comparison for Target: {target_name} (Error bars = ±1 std dev)'
            plt.title(title, fontweight='bold', fontsize=14)
            plt.xticks(index + bar_width * (len(metric_names) - 1) / 2, plot_model_names, rotation=15, ha="right")
            plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            summary_plot_path = os.path.join(summary_plot_dir, f"summary_target_{target_name}.png")
            try:
                plt.savefig(summary_plot_path)
                print(f"Saved target-specific summary plot: {summary_plot_path}")
            except Exception as e:
                print(f"Warning: Could not save target-specific summary plot {summary_plot_path}: {e}")
        else:
            print(f"Skipping target-specific summary plot for {target_name} as no valid data was found.")
        
        plt.close()

    # --- Prepare data (mean and std) for processing ---
    # Filter models that ran successfully AND are in the desired order list
    available_models_full_unsorted = [m for m in valid_metrics.keys() if m in full_model_order]
    available_models_full = sorted(available_models_full_unsorted, key=lambda m: full_model_order.index(m))

    # Prepare dictionaries to hold mean and std dev data, structured by [metric][target]
    summary_data_mean = {metric: {target: [] for target in target_names} for metric in metric_names}
    summary_data_std = {metric: {target: [] for target in target_names} for metric in metric_names}
    # Prepare data for CSV export (rows: model, columns: multi-index (Target, Metric, Stat))
    csv_data = {} 

    for model_name in available_models_full: # Iterate in the desired full order
        metrics_data = valid_metrics[model_name]
        is_aggregated = isinstance(metrics_data, dict) and 'mean' in metrics_data
        model_csv_row = {} # Store data for this model's row in CSV

        for metric in metric_names:
            for target in target_names:
                if is_aggregated:
                    mean_val = metrics_data.get('mean', {}).get(target, {}).get(metric, np.nan)
                    std_val = metrics_data.get('std', {}).get(target, {}).get(metric, np.nan)
                else: # Handle non-aggregated (single-run) models
                    mean_val = metrics_data.get(target, {}).get(metric, np.nan)
                    std_val = 0 # No std dev for single run
                
                # Append data for plotting
                summary_data_mean[metric][target].append(mean_val)
                summary_data_std[metric][target].append(std_val)
                
                # Add data for CSV
                model_csv_row[(target, f'{metric}_mean')] = mean_val
                model_csv_row[(target, f'{metric}_std')] = std_val if std_val != 0 else np.nan # Use NaN for 0 std in CSV for clarity

        csv_data[model_name] = model_csv_row

    # --- Generate Overall Plots ---
    generate_summary_plot(available_models_full, summary_data_mean, summary_data_std,
                          "Across Models", "summary_")

    # --- Prepare data and generate SUBSET plot --- (Reuses full data, just filters model names)
    available_models_subset = [m for m in available_models_full if m in subset_model_order] # Filter from already ordered list
    # Re-index the full data dictionaries for the subset of models
    subset_indices = [available_models_full.index(m) for m in available_models_subset]
    summary_data_mean_subset = {metric: {target: [summary_data_mean[metric][target][idx] for idx in subset_indices] 
                                        for target in target_names} 
                              for metric in metric_names}
    summary_data_std_subset = {metric: {target: [summary_data_std[metric][target][idx] for idx in subset_indices] 
                                       for target in target_names} 
                             for metric in metric_names}
    generate_summary_plot(available_models_subset, summary_data_mean_subset, summary_data_std_subset,
                          "Across Subset Models", "summary_subset_")
                          
    # --- Generate Target-Specific Plots --- 
    for target in target_names:
         generate_target_specific_plot(target, available_models_full, summary_data_mean, summary_data_std)
         
    # --- Save Summary Metrics to CSV --- 
    try:
        # Create MultiIndex DataFrame
        csv_df = pd.DataFrame.from_dict(csv_data, orient='index')
        csv_df.columns = pd.MultiIndex.from_tuples(csv_df.columns, names=['Target', 'Metric'])
        csv_df = csv_df.sort_index(axis=1) # Sort columns for consistency
        csv_df.index.name = 'Model'
        
        csv_path = os.path.join(summary_plot_dir, "summary_metrics.csv")
        csv_df.to_csv(csv_path)
        print(f"\nSaved summary metrics to: {csv_path}")
        print(csv_df)
    except Exception as e:
        print(f"\nWarning: Could not save summary metrics to CSV: {e}") 