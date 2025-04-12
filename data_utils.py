import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

print("Initializing Data Utils Module...")

# --- Data Loading and Preprocessing ---
def load_molecular_data(excel_path, sheet_name, targets):
    """
    Loads and preprocesses molecular data from an Excel file.
    Returns Tensors for features/targets (except Y_test which is numpy).
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
                     if pd.isna(median_val):
                          median_val = 0 # Fallback if entire column is NaN
                     input_df[col] = input_df[col].fillna(median_val)


        X, Y = input_df.values, df_clean[targets].values

        x_nan_count = np.isnan(X).sum()
        y_nan_count = np.isnan(Y).sum()
        if x_nan_count > 0 or y_nan_count > 0:
            print(f"Warning: Found {x_nan_count} NaN in X, {y_nan_count} in Y after imputation.")

        # Clipping before replacing inf values
        try:
            X = np.clip(X, np.nanquantile(X, 0.001), np.nanquantile(X, 0.999))
            Y = np.clip(Y, np.nanquantile(Y, 0.001), np.nanquantile(Y, 0.999))
        except Exception as e:
            print(f"Warning: Could not apply quantile clipping: {e}")

        # Replace inf with large numbers, then handle remaining NaNs
        X = np.nan_to_num(X, nan=np.nan, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        Y = np.nan_to_num(Y, nan=np.nan, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        X = np.nan_to_num(X, nan=0.0) # Replace remaining NaNs (from clipping potentially)
        Y = np.nan_to_num(Y, nan=0.0)
        print(f"Final data shapes after clipping/nan_to_num: X: {X.shape}, Y: {Y.shape}")


        stratify = None
        if len(X) > 10:
            try:
                y_mean = np.mean(Y, axis=1)
                n_bins = min(5, len(X) // 5)
                if n_bins > 1:
                    y_strat = pd.qcut(y_mean, n_bins, labels=False, duplicates='drop')
                    # Check if stratification resulted in only one unique value
                    if len(set(y_strat)) > 1:
                         stratify = y_strat
                    else:
                         print("Warning: Stratification resulted in a single bin, disabling stratification.")
            except Exception as e:
                print(f"Warning: Stratification failed: {e}")


        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=stratify
        )

        # Initialize scalers
        scaler_x = RobustScaler()
        scaler_y = RobustScaler()

        # Fit and transform training data, transform test data
        try:
            X_train_scaled = scaler_x.fit_transform(X_train)
            Y_train_scaled = scaler_y.fit_transform(Y_train)
            X_test_scaled = scaler_x.transform(X_test)
        except ValueError as e:
            print(f"Warning: RobustScaler failed with ValueError: {e}. Falling back to StandardScaler.")
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            X_train_scaled = scaler_x.fit_transform(X_train)
            Y_train_scaled = scaler_y.fit_transform(Y_train)
            X_test_scaled = scaler_x.transform(X_test)
        except Exception as e: # Catch other potential errors
             print(f"Warning: Scaler failed with unexpected error: {e}. Falling back to StandardScaler.")
             scaler_x = StandardScaler()
             scaler_y = StandardScaler()
             X_train_scaled = scaler_x.fit_transform(X_train)
             Y_train_scaled = scaler_y.fit_transform(Y_train)
             X_test_scaled = scaler_x.transform(X_test)
        
        # Keep Y_test unscaled for evaluation
        Y_test_numpy = Y_test.copy() # Original Y_test data

        # Final check for NaN/Inf after scaling
        for arr, name in [(X_train_scaled, "X_train_scaled"), (Y_train_scaled, "Y_train_scaled"),
                          (X_test_scaled, "X_test_scaled")]:
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"Warning: Found NaN/Inf in {name} after scaling! Replacing with zeros.")
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert scaled data to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        print("Data processing completed successfully.")
        # Return tensors for DL and numpy Y_test for evaluation
        return (X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_numpy,
                scaler_y, input_df.columns.tolist(), targets)

    except FileNotFoundError:
         print(f"Error: Excel file not found at {excel_path}")
         raise # Re-raise error after logging
    except Exception as e:
        print(f"Error in data loading: {e}")
        import traceback
        traceback.print_exc()
        raise # Re-raise error after logging 