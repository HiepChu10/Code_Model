# XGBoost Cryptocurrency Prediction Pipeline (Memory Optimized)

A modular machine learning pipeline for predicting cryptocurrency price movements using XGBoost with **advanced memory optimization** for large datasets (3M+ records).

## 🚀 Key Features

- **Memory Optimization**: Handles large datasets (3M+ records) with intelligent memory management
- **Float32 Optimization**: 50% memory savings by using float32 instead of float64
- **Automatic Sampling**: Falls back to data sampling when memory is insufficient
- **Chunked Loading**: Loads data in chunks for memory-constrained systems
- **Batch Processing**: Processes predictions in batches for large datasets
- **Real-time Memory Monitoring**: Tracks memory usage with warnings and cleanup
- **Error Recovery**: Graceful handling of memory errors with fallback strategies
- **Modular Architecture**: Clean separation of concerns for easy debugging

## 💾 Memory Requirements

### For Small Datasets (< 500K records)
- **RAM**: 8GB minimum
- **Recommended**: 16GB

### For Large Datasets (3M+ records)  
- **RAM**: 16GB minimum, 32GB+ recommended
- **Enable sampling**: Set `USE_SAMPLING = True` in config
- **Float32 savings**: ~50% memory reduction

## 📁 Cấu trúc thư mục

```
├── main.py              # File chính để chạy toàn bộ pipeline
├── config.py            # Cấu hình và settings
├── data_handler.py      # Xử lý dữ liệu
├── model_trainer.py     # Training XGBoost model
├── evaluator.py         # Đánh giá model và tạo plots
├── report_generator.py  # Tạo báo cáo Word
├── utils.py             # Các hàm tiện ích
├── README.md            # Hướng dẫn sử dụng
└── output/              # Thư mục kết quả (được tạo tự động)
```

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies

```bash
# Install from requirements.txt (recommended)
pip install -r requirements.txt

# Or install manually
pip install pandas numpy xgboost scikit-learn matplotlib seaborn python-docx psutil
```

### 2. Cấu hình

Chỉnh sửa file `config.py`:

```python
# Đường dẫn dữ liệu
DATA_PATH = 'path/to/your/training_data.csv'
TEST_NEW_PATH = 'path/to/your/test_data.csv'

# Memory optimization settings - AGGRESSIVE FLOAT32 MODE
MEMORY_SETTINGS = {
    'CHUNK_SIZE': 100000,      # Process data in chunks
    'USE_SAMPLING': False,     # Set to True if memory issues persist
    'SAMPLE_SIZE': 500000,     # Sample size if sampling is enabled
    'OPTIMIZE_DTYPES': True,   # Optimize data types (FLOAT32)
    'CLEAR_MEMORY': True,      # Clear memory after each step
}

# For datasets with 3M+ records, use aggressive settings:
# MEMORY_SETTINGS = {
#     'USE_SAMPLING': True,     # Enable sampling
#     'SAMPLE_SIZE': 300000,    # Reduce sample size  
#     'CHUNK_SIZE': 50000,      # Smaller chunks
# }

# XGBoost parameters - Memory optimized
XGB_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'random_state': 42,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'n_jobs': 4  # Reduced from 20 to save memory
}
```

### 3. Chạy pipeline

```bash
python main.py
```

## 📦 Mô tả các module

### `config.py`
- Chứa tất cả cấu hình, đường dẫn file, settings thử nghiệm
- Tham số XGBoost

### `data_handler.py`
- **DataHandler class**: Quản lý việc đọc và chuẩn bị dữ liệu
- `load_main_data()`: Đọc dữ liệu training chính
- `split_data()`: Chia train/test
- `load_new_test_data()`: Đọc dữ liệu test bên ngoài

### `model_trainer.py`
- **ModelTrainer class**: Quản lý việc training model
- `train_model()`: Train XGBoost với scale_pos_weight tự động
- `calculate_scale_pos_weight()`: Tính trọng số cân bằng dữ liệu

### `evaluator.py`
- **ModelEvaluator class**: Đánh giá model và tạo visualizations
- `evaluate_and_plot()`: Tạo classification report, confusion matrix, prediction charts
- `plot_feature_importance()`: Biểu đồ tầm quan trọng của features

### `report_generator.py`
- **ReportGenerator class**: Tạo báo cáo Word tự động
- `generate_word_report()`: Tạo file .docx với format chuẩn
- Bao gồm tất cả tables, charts, và thống kê

### `utils.py`
- Các hàm tiện ích: `format_int()`, `format_percentage()`
- `setup_logging()`: Thiết lập logging
- `create_output_dir()`: Tạo thư mục output

### `main.py`
- Điều phối toàn bộ pipeline
- Xử lý lỗi và logging
- Chạy tất cả experiments tự động

## 📊 Kết quả

Mỗi experiment sẽ tạo ra:

```
output/
└── {Y_TYPE}_thr{threshold}/
    └── featureset_{n}/
        ├── log.txt                          # Log file
        ├── report_test.csv                  # Classification report (test)
        ├── report_new.csv                   # Classification report (new data)
        ├── prediction_bar_test.png          # Prediction charts
        ├── confusion_matrix_test.png        # Confusion matrices
        ├── feature_importance.png           # Feature importance
        └── Report-xxxxx-Hiep-{TOKEN}-XGB-{DATE}.docx  # Word report
```

## 🐛 Debug

Để debug specific component:

```python
# Test data loading
from data_handler import DataHandler
data_handler = DataHandler('path/to/data.csv', 'path/to/test.csv')

# Test model training
from model_trainer import ModelTrainer
trainer = ModelTrainer(XGB_PARAMS)

# Test evaluation
from evaluator import ModelEvaluator
evaluator = ModelEvaluator('output_dir')
```

## 🔧 Memory Error Troubleshooting

### "Unable to allocate X.XX GiB for an array" Error

This error occurs when your dataset is too large for available memory. Here are solutions:

#### Option 1: Enable Sampling (Quickest Fix)
```python
# In config.py, change:
MEMORY_SETTINGS = {
    'USE_SAMPLING': True,      # Enable sampling
    'SAMPLE_SIZE': 300000,     # Reduce this number if still failing
    'OPTIMIZE_DTYPES': True,   # Keep float32 optimization
    'CLEAR_MEMORY': True,
}
```

#### Option 2: Reduce Chunk Size
```python
# For very limited memory systems
MEMORY_SETTINGS = {
    'CHUNK_SIZE': 50000,       # Smaller chunks
    'USE_SAMPLING': True,
    'SAMPLE_SIZE': 200000,     # Even smaller sample
}
```

#### Option 3: System-level Solutions
- **Close other applications** to free RAM
- **Increase virtual memory/swap space**
- **Upgrade to more RAM** (recommended for 3M+ records)

### Memory Optimization Features

1. **Float32 Conversion**: Automatically reduces memory by 50%
2. **Automatic Sampling**: Activates when memory usage > 80%
3. **Chunked Loading**: Processes large files incrementally
4. **Memory Monitoring**: Real-time tracking with warnings
5. **Garbage Collection**: Aggressive cleanup between steps

### Expected Memory Usage (3.5M records, 17 features)

| Mode | Memory Usage | Accuracy Impact |
|------|-------------|----------------|
| Float64 (original) | ~508MB | Baseline |
| Float32 (optimized) | ~254MB | Negligible |
| Sampled (300K) | ~22MB | Minimal |

## 📋 Requirements

- Python 3.7+
- pandas>=1.3.0
- numpy>=1.20.0
- xgboost>=1.5.0
- scikit-learn>=1.0.0
- matplotlib>=3.3.0
- seaborn>=0.11.0
- python-docx>=0.8.11
- psutil>=5.8.0 (for memory monitoring)

## 🔧 Customization

1. **Thêm features mới**: Chỉnh sửa `feature_sets` trong `config.py`
2. **Thay đổi model parameters**: Chỉnh sửa `XGB_PARAMS` trong `config.py`
3. **Memory settings**: Điều chỉnh `MEMORY_SETTINGS` trong `config.py`
4. **Custom evaluation metrics**: Mở rộng `ModelEvaluator` class
5. **Custom report format**: Chỉnh sửa `ReportGenerator` class