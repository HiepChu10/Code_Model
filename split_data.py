import pandas as pd
import os
from tqdm import tqdm

# ================================
# CẤU HÌNH NGƯỜI DÙNG
# ================================
FILE_PREFIX = "TRX"                                             # Biến có thể thay đổi
input_file = f"data/{FILE_PREFIX}/TRX_2025-06-13_03-54-57.csv"  # Đường dẫn file CSV
output_dir = f"data/{FILE_PREFIX}/output"                       # Thư mục lưu kết quả                        

print(f"Đang xử lý file: {input_file}")
print(f"Prefix cho tên file đầu ra: {FILE_PREFIX}")

# ================================
# Đọc dữ liệu từ file
# ================================
print("Đang đọc dữ liệu từ file CSV...")
df = pd.read_csv(input_file, parse_dates=['OpenTime'])
print("Hoàn tất đọc dữ liệu.")

# ================================
# Kiểm tra cột 'StartCumulative'
# ================================
print("Đang kiểm tra cột 'StartCumulative'...")
if 'StartCumulative' not in df.columns:
    raise ValueError("Cột 'StartCumulative' không tồn tại trong file CSV!")
print("Cột 'StartCumulative' đã tồn tại.")

# ================================
# Tạo thư mục output nếu chưa có
# ================================
print(f"Đang kiểm tra hoặc tạo thư mục output: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
print("Thư mục output đã sẵn sàng.")

# ================================
# Tách và lưu từng nhóm
# ================================
print("Bắt đầu tách và lưu từng nhóm theo StartCumulative...")

for i, (start_cum, group) in enumerate(tqdm(df.groupby('StartCumulative'), desc="Đang xử lý nhóm"), start=1):
    group_sorted = group.sort_values('OpenTime')
    first_time = group_sorted.iloc[0]['OpenTime']
    formatted_time = first_time.strftime('%Y%m%d_%H%M%S')
    record_count = len(group_sorted)

    # Tạo tên file theo format yêu cầu
    file_prefix = f"{i:02d}"
    output_filename = f"{FILE_PREFIX}_{file_prefix}_{start_cum}_{formatted_time}_{record_count}.csv"
    output_path = os.path.join(output_dir, output_filename)

    # Ghi file CSV
    group_sorted.to_csv(output_path, index=False)
    print(f"\nĐã lưu: {output_filename}")

print("\nHoàn tất quá trình tách và lưu file.")
