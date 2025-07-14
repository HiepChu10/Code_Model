# ===================== CONFIG =====================
# DATA_PATH = '/home/shared_data/SOL/SOL_2025-03-01_00-00-00.csv'
# TEST_NEW_PATH = '/home/shared_data/SOL/SOL_2025-07-04_07-46-13.csv'
DATA_PATH = '/home/shared_data/TRX/BINANCE/TRX_2025-03-01_00-00-00.csv'
TEST_NEW_PATH = '/home/shared_data/TRX/BINANCE/TRX_2025-07-11_03-41-28.csv'
# DATA_PATH = '/home/shared_data/ETH/ETH_2025-03-01_00-00-00.csv'
# TEST_NEW_PATH = '/home/shared_data/ETH/ETH_2025-07-04_08-08-16.csv'

CURRENT_REPORT_NUMBER = 471
DOCX_SAVE_PATH = r"/home/ai2/report"

settings = [
    # {'Y_TYPE': 'close_vs_maxhigh120_r10', 'threshold': 0.4},
    # {'Y_TYPE': 'close_vs_minlow120_r10', 'threshold': 0.4},
    # {'Y_TYPE': 'close_vs_maxhigh120_r20', 'threshold': 0.4},
    # {'Y_TYPE': 'close_vs_minlow120_r20', 'threshold': 0.4},
    # {'Y_TYPE': 'close_vs_maxhigh120_r10', 'threshold': 0.5},
    # {'Y_TYPE': 'close_vs_minlow120_r10', 'threshold': 0.5},
    # {'Y_TYPE': 'close_vs_maxhigh120_r20', 'threshold': 0.5},
    # {'Y_TYPE': 'close_vs_minlow120_r20', 'threshold': 0.5},
    # {'Y_TYPE': 'close_vs_maxhigh120_r30', 'threshold': 0.6},
    # {'Y_TYPE': 'close_vs_minlow120_r30', 'threshold': 0.6},
    {'Y_TYPE': 'close_vs_maxhigh120_r1000', 'threshold': 0.8},
    {'Y_TYPE': 'close_vs_minlow120_r1000', 'threshold': 0.8},
    # {'Y_TYPE': 'close_vs_maxhigh120_r10', 'threshold': 0.8},
    # {'Y_TYPE': 'close_vs_minlow120_r10', 'threshold': 0.8},
    # {'Y_TYPE': 'close_vs_maxhigh120_r20', 'threshold': 0.8},
    # {'Y_TYPE': 'close_vs_minlow120_r20', 'threshold': 0.8},
    # {'Y_TYPE': 'close_vs_maxhigh120_r10', 'threshold': 0.9},
    # {'Y_TYPE': 'close_vs_minlow120_r10', 'threshold': 0.9},
    # {'Y_TYPE': 'close_vs_maxhigh120_r20', 'threshold': 0.9}
    # {'Y_TYPE': 'close_vs_minlow120_r20', 'threshold': 0.9},
]

feature_sets = [
    [
        'sma_macd_diff_5', 'sma_macd_diff_10', 'sma_macd_diff_sub', 'macd_diff', 'macd',
        'close_div_ema9', 'close_div_ema25', 'close_div_ema50',
        'close_div_sma9', 'close_div_sma25', 'close_div_sma50',
        'close_div_ema100', 'close_div_ema200', 'close_div_sma100', 'close_div_sma200',
        'close_div_ema300', 'close_div_ema500', 'close_div_ema1000',
        'close_div_sma300', 'close_div_sma500', 'close_div_sma1000'
    ],
    [
        'sma_macd_diff_5', 'sma_macd_diff_10', 'sma_macd_diff_sub', 'macd_diff', 'macd',
        'close_div_ema9', 'close_div_ema25', 'close_div_ema50',
        'close_div_sma9', 'close_div_sma25', 'close_div_sma50',
        'close_div_ema100', 'close_div_ema200', 'close_div_sma100', 'close_div_sma200',
        'volume_usdt', 'volume_usdt_10', 'volume_usdt_60',
        'close_div_ema300', 'close_div_ema500', 'close_div_ema1000',
        'close_div_sma300', 'close_div_sma500', 'close_div_sma1000'
    ]
]

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
    'n_jobs': 7  # Single thread for maximum memory stability
}

# Memory optimization settings - AGGRESSIVE FLOAT32 MODE
# MEMORY_SETTINGS = {
#     'CHUNK_SIZE': 100000,      # Process data in chunks
#     'OPTIMIZE_DTYPES': True,   # Optimize data types to save memory (FLOAT32)
#     'CLEAR_MEMORY': True,      # Clear memory after each step
# }

# Memory optimization settings
MEMORY_SETTINGS = {
    'CHUNK_SIZE': 100000,      # Process data in chunks when memory insufficient
    'OPTIMIZE_DTYPES': True,   # Float32 optimization to save memory
    'CLEAR_MEMORY': True,      # Clear memory after each step
}

# Mapping r-value  →  rate
RATE_MAP = {
    8:    0.08,
    10:   0.1,
    20:   0.2,
    30:   0.3,
    40:   0.4,
    50:   0.5,
    100:  1.0,
    200:  2.0,
    300:  3.0,
    500:  5.0,
    1000: 10.0,
} 