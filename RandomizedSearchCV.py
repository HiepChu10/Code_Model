from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd


# Gợi ý cho tập lớn và mất cân bằng
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [100, 150, 200, 250],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [10, 20],
    'max_leaf_nodes': [31, 63],
    'l2_regularization': [0.0, 1.0],
    'early_stopping': [True],
    'validation_fraction': [0.1]
}
# Đường dẫn
data_path = 'C:/Users/ai1/Desktop/Model/data/TRX/output/TRX_28_27994440_20230324_144900_1168569.csv'

# Đọc dữ liệu
df = pd.read_csv(data_path)

# Chuyển đổi OpenTime thành datetime
df['OpenTime'] = pd.to_datetime(df['OpenTime'])

df.set_index('OpenTime', inplace=True)

Y_TYPE = 'close_vs_maxhigh5_r8'

model = HistGradientBoostingClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,
    # scoring='f1',  # ⚠️ hoặc 'roc_auc' nếu bạn quan tâm phân biệt đúng lớp 1
    scoring='roc_auc', 
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Best Params:", random_search.best_params_)
