import os
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === 1. Khởi tạo mô hình và bộ tham số ===
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [10, 20],
    'max_leaf_nodes': [31, 63],
    'l2_regularization': [0.0, 1.0],
    'early_stopping': [True],
    'validation_fraction': [0.1]
}

model = HistGradientBoostingClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1',  # bạn có thể dùng 'roc_auc' nếu phù hợp hơn
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# === 2. Huấn luyện mô hình ===
random_search.fit(X_train_smote, y_train_smote)
best_model = random_search.best_estimator_

# === 3. Đánh giá trên tập validation ===
y_val_pred = best_model.predict(X_val)
print("Validation Report:\n", classification_report(y_val, y_val_pred, digits=4))

# Confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(best_model, X_val, y_val, cmap='Blues')
disp.ax_.set_title("Validation Confusion Matrix")
plt.show()

# === 4. Lưu mô hình ===
# Ví dụ: nếu bạn đọc dữ liệu từ file "data/stock_A.csv"
data_file_path = "data/stock_A.csv"
dataset_name = os.path.splitext(os.path.basename(data_file_path))[0]  # => "stock_A"

output_dir = f"models/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, f"gradient_boosting_{Y_TYPE}.joblib")
joblib.dump(best_model, model_path)

print(f"✅ Mô hình đã lưu tại: {model_path}")
