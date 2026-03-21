import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_evaluate_svr(data_path, results_dir="reports"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 1. Split Data
    # For SVR, training on 35k rows is computationally heavy O(N^3).
    # We will use exactly 1 year of data (8760 hours) for training
    # and 3 months (2160 hours) for testing to evaluate our baseline.
    train_size = 8760
    test_size = 2160
    
    train_df = df.iloc[:train_size] 
    test_df = df.iloc[train_size:train_size+test_size]
    
    X_cols = ['hour', 'day_of_week', 'month', 'day_of_year', 
              'lag_1h', 'lag_2h', 'lag_24h', 'rolling_mean_24h', 'rolling_std_24h']
    y_col = 'MW_Load'
    
    X_train, y_train = train_df[X_cols], train_df[y_col]
    X_test, y_test = test_df[X_cols], test_df[y_col]
    
    # 2. Scaling
    # Support Vector Machines are extremely sensitive to unscaled data
    print("Scaling temporal and lag features...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    # 3. Model Definition and Training
    # SVR with RBF kernel handles non-linear relationships smoothly
    print("Training Support Vector Regressor (RBF Kernel)...")
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train_scaled)
    
    # 4. Evaluation and Validation
    print("Evaluating Model A (SVR Baseline)...")
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Re-transform predictions to original MW Load scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_values = y_test.values
    
    mae = mean_absolute_error(y_test_values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_values, y_pred))
    
    print("\n" + "="*40)
    print("--- Model A (Advanced ML) Validation ---")
    print("="*40)
    print(f"Algorithm : Support Vector Regressor (RBF)")
    print(f"Test Size : {test_size} hours (3 months)")
    print(f"MAE       : {mae:.2f} MW")
    print(f"RMSE      : {rmse:.2f} MW")
    print("========================================")
    
    # 5. Visualization Delivery (Required for architecture and report)
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # Plotting exactly 2 weeks (336 hours) of data to see cycle fit
    weeks = 336 
    plt.plot(test_df.index[:weeks], y_test_values[:weeks], label="Actual Load", color='black', alpha=0.7)
    plt.plot(test_df.index[:weeks], y_pred[:weeks], label="SVR Prediction", color='red', linestyle='dashed', linewidth=2)
    
    plt.title("SVR Baseline (RBF Kernel) vs Actual Energy Demand (2 Weeks)")
    plt.xlabel("Datetime")
    plt.ylabel("Demand (MW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, "model_a_svr_forecast.png")
    plt.savefig(plot_path)
    print(f"Saved forecast visualization to: {plot_path}")

if __name__ == "__main__":
    train_and_evaluate_svr("data/processed/featured_data.csv")
