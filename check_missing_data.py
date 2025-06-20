# ================================ READ ME! ================================
#           Thay đổi đường dẫn và tên thư mục trước khi sử dụng


import pandas as pd
from tqdm import tqdm

# ============================
# Đường dẫn file
# ============================
FILE_PREFIX = "TRX"   
file_path = f"data/{FILE_PREFIX}/TRX_2025-06-13_03-54-57.csv"  # Đổi đường dẫn

print(f"\nĐang xử lý file: {file_path}")

# ============================
# Đọc dữ liệu, parse OpenTime
# ============================
print("Đang đọc dữ liệu và phân tích cột OpenTime...")
df = pd.read_csv(file_path, parse_dates=['OpenTime'])
print("Đã đọc xong dữ liệu.")

# ============================
# Chuẩn hóa OpenTime
# ============================
print("Chuẩn hóa OpenTime về đúng giây...")
df['OpenTime'] = df['OpenTime'].dt.floor('s')
print("Đã chuẩn hóa thời gian.")

# ============================
# Kiểm tra thời điểm trùng lặp
# ============================
print("Kiểm tra các thời điểm OpenTime bị trùng...")
duplicated_times = df[df.duplicated(subset=['OpenTime'], keep=False)]
duplicated_counts = duplicated_times['OpenTime'].value_counts()

if not duplicated_counts.empty:
    print("10 thời điểm OpenTime bị trùng đầu tiên:")
    print(duplicated_counts.head(10))
else:
    print("Không có thời điểm nào bị trùng OpenTime.")

# ============================
# Loại bỏ dòng trùng
# ============================
print("Loại bỏ các dòng trùng theo OpenTime...")
df_unique = df.drop_duplicates(subset=['OpenTime'])

# ============================
# Sắp xếp theo thời gian
# ============================
print("Sắp xếp dữ liệu theo thời gian...")
df_sorted = df_unique.sort_values('OpenTime').reset_index(drop=True)

# ============================
# Tính độ chênh lệch thời gian
# ============================
print("Tính độ chênh lệch thời gian giữa các dòng...")
df_sorted['diff'] = df_sorted['OpenTime'].diff()

# ============================
# Kiểm tra đứt gãy
# ============================
print("Đang kiểm tra các khoảng đứt gãy > 1 phút...")
gap_indices = []
for i in tqdm(range(1, len(df_sorted)), desc="Kiểm tra khoảng đứt gãy"):
    if df_sorted.loc[i, 'diff'] > pd.Timedelta(minutes=1):
        gap_indices.append(i)

# ============================
# In kết quả đứt gãy
# ============================
if gap_indices:
    print(f"\nPhát hiện {len(gap_indices)} khoảng đứt gãy (>1 phút). Thời điểm bắt đầu đứt:")
    for idx in gap_indices:
        print(df_sorted.loc[idx - 1, 'OpenTime'])
else:
    print("\nKhông phát hiện khoảng đứt gãy.")

