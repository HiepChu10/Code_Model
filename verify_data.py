import pandas as pd
import numpy as np
import clickhouse_connect
from tqdm import tqdm

# ============================
# THIẾT LẬP BIẾN TÙY CHỈNH
# ============================
CLICKHOUSE_CONFIG = {
    'host': '117.0.34.78',
    'port': 8123,
    'username': 'looker',
    'password': ''
}

FILE_PREFIX = "TRX"  
START_TIME = "2022-05-11 09:00:00.000"              # Thay đổi 
END_TIME = "2025-03-21 19:50:00.000"                # Thay đổi 
CSV_PATH = f"data/{FILE_PREFIX}/TRX_2025-06-13_03-54-57.csv"       # Thay đổi 
OPEN_TIME_COL = "OpenTime"      
FLOAT_TOLERANCE = 1e-6

# ============================
# HIỂN THỊ THÔNG TIN CHUNG
# ============================
print(f"Đang xử lý khoảng thời gian: {START_TIME} → {END_TIME}")
print(f"Đang đọc file CSV: {CSV_PATH}")

# ============================
# TRUY VẤN DỮ LIỆU
# ============================
print("\nĐang truy vấn dữ liệu từ ClickHouse...")

QUERY_A = f"""
    SELECT * 
    FROM binance_data_indicator.trxusdt_1m_forward_ohlc FINAL
    WHERE OpenTime >= '{START_TIME}' AND OpenTime <= '{END_TIME}'
    ORDER BY OpenTime ASC
    LIMIT 30000
"""

QUERY_B = f"""
    SELECT * 
    FROM binance_data_indicator.trxusdt_1m_extended FINAL
    WHERE OpenTime >= '{START_TIME}' AND OpenTime <= '{END_TIME}'
    ORDER BY OpenTime ASC
    LIMIT 30000
"""

client = clickhouse_connect.get_client(**CLICKHOUSE_CONFIG)

with tqdm(total=2, desc="Truy vấn ClickHouse", bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}") as pbar:
    df_a = client.query_df(QUERY_A)
    pbar.update(1)
    df_b = client.query_df(QUERY_B)
    pbar.update(1)

# ============================
# ĐỌC FILE CSV
# ============================
print("\nĐang đọc file CSV...")

with tqdm(total=1, desc="Đọc CSV", bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}") as pbar:
    df_c = pd.read_csv(CSV_PATH)
    pbar.update(1)

# ============================
# CHUẨN HÓA DỮ LIỆU & LỌC CSV
# ============================
print("\nĐang chuẩn hóa dữ liệu...")

df_c[OPEN_TIME_COL] = pd.to_datetime(df_c[OPEN_TIME_COL])
df_a[OPEN_TIME_COL] = pd.to_datetime(df_a[OPEN_TIME_COL])
df_b[OPEN_TIME_COL] = pd.to_datetime(df_b[OPEN_TIME_COL])

start_dt = pd.to_datetime(START_TIME)
end_dt = pd.to_datetime(END_TIME)
df_c = df_c[(df_c[OPEN_TIME_COL] >= start_dt) & (df_c[OPEN_TIME_COL] <= end_dt)]

# ============================
# GHÉP DỮ LIỆU
# ============================
print("\nĐang ghép dữ liệu theo OpenTime...")

merged_a = df_c.merge(df_a, on=OPEN_TIME_COL, suffixes=('_C', '_A'))
merged_b = df_c.merge(df_b, on=OPEN_TIME_COL, suffixes=('_C', '_B'))

# ============================
# HÀM SO SÁNH
# ============================
def compare_columns(df, suffix_1, suffix_2, atol=FLOAT_TOLERANCE):
    diffs = {}
    compare_cols = [col for col in df.columns if col.endswith(suffix_1)]
    for col in tqdm(compare_cols, desc=f"So sánh {suffix_1} vs {suffix_2}"):
        base_col = col[:-len(suffix_1)]
        col1 = col
        col2 = base_col + suffix_2
        if col2 in df.columns:
            series1 = df[col1]
            series2 = df[col2]
            if np.issubdtype(series1.dtype, np.number):
                is_diff = ~np.isclose(series1, series2, equal_nan=True, atol=atol)
            else:
                is_diff = series1 != series2
            diff_df = df[is_diff][[OPEN_TIME_COL, col1, col2]]
            if not diff_df.empty:
                diffs[base_col] = diff_df
    return diffs

# ============================
# SO SÁNH DỮ LIỆU
# ============================
print("\nSo sánh dữ liệu giữa bảng C và bảng A...")
diffs_a = compare_columns(merged_a, '_C', '_A')

print("\nSo sánh dữ liệu giữa bảng C và bảng B...")
diffs_b = compare_columns(merged_b, '_C', '_B')

# ============================
# IN KẾT QUẢ
# ============================
def print_diffs(title, diffs):
    print(f"\n===== {title} =====")
    if not diffs:
        print("Không phát hiện sai lệch.")
    else:
        for col, diff_df in diffs.items():
            print(f"\nCột: {col}")
            print(diff_df)
            print("\nCó sai lệch.")

print_diffs("Sai lệch giữa bảng C và bảng A", diffs_a)
print_diffs("Sai lệch giữa bảng C và bảng B", diffs_b)
