from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, precision_score
from datetime import datetime
import pandas as pd
import os
import joblib
import time
import warnings
warnings.filterwarnings("ignore")

# Đường dẫn
data_path = '/home/shared_data/BTC/BTC_2025-03-01_00-00-00.csv'

# Tên dataset & ngày hiện tại
dataset_name = os.path.splitext(os.path.basename(data_path))[0]
today_str = datetime.today().strftime('%Y-%m-%d')

# Đọc dữ liệu
df = pd.read_csv(data_path)
df['OpenTime'] = pd.to_datetime(df['OpenTime'])
df.set_index('OpenTime', inplace=True)

# Biến mục tiêu
Y_TYPE = 'close_vs_minlow60_r10'

# Tập feature
features = [
    'sma_macd_diff_5',
    'sma_macd_diff_10', 
    'sma_macd_diff_sub',
    'macd_diff',
    'macd',
    'close_div_ema9',
    'close_div_ema25',
    'close_div_ema50',
    'close_div_sma9',
    'close_div_sma25',
    'close_div_sma50'
]

X = df[features]
y = df[Y_TYPE]

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

# Tính scale_pos_weight
import numpy as np
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# Scorer: Precision cho nhãn 1
scorer = make_scorer(precision_score, pos_label=1)

# XGBoost model
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=20,
)

# Grid các siêu tham số
param_grid = {
    'max_depth': [6, 8, 10],
    'min_child_weight': [10, 15, 20],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [300, 500],
    'reg_alpha': [0.5, 1],
    'reg_lambda': [0.5, 1],
    'scale_pos_weight': [scale_pos_weight]
}

# GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=scorer,
    cv=3,
    verbose=3,  # HIỂN THỊ tiến trình trên console
    n_jobs=20
)

# Bắt đầu tính thời gian
print("Bắt đầu Grid Search...")
print(f"Tổng số combinations: {len(param_grid['max_depth']) * len(param_grid['min_child_weight']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['learning_rate']) * len(param_grid['n_estimators']) * len(param_grid['reg_lambda'])}")
start_time = time.time()

# Chạy tìm kiếm
print("Đang chạy GridSearchCV...")
grid_search.fit(X_train, y_train)

# Tính thời gian kết thúc
end_time = time.time()
elapsed_time = end_time - start_time

# In kết quả
print("\n" + "="*50)
print("KẾT QUẢ GRID SEARCH")
print("="*50)
print(f"Thời gian chạy: {elapsed_time:.2f} giây ({elapsed_time/60:.2f} phút)")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Precision for label 1: {grid_search.best_score_:.4f}")
print("="*50)

# Lưu mô hình tốt nhất
best_model = grid_search.best_estimator_

model_save_path = f"best_xgb_model.joblib"
joblib.dump(best_model, model_save_path)
print(f"Best model saved to: {model_save_path}")
print(f"Hoàn thành! Tổng thời gian: {elapsed_time:.2f} giây")

