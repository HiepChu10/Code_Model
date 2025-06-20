# ================================ READ ME! ================================
#           Thay đổi tên bảng và tên thư mục trước khi sử dụng

import pandas as pd
from clickhouse_connect import get_client
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

# --- KẾT NỐI CLICKHOUSE ---
CLICKHOUSE_HOST = '117.0.34.78'
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = 'looker'
CLICKHOUSE_PASSWORD = ''
DATABASE_NAME = 'binance_data_indicator'

# --- CẤU HÌNH TRUY VẤN ---
TABLE_A = 'trxusdt_1m_extended'         # Đổi tên bảng
TABLE_B = 'trxusdt_1m_forward_ohlc'     # Đổi tên bảng

# --- THỜI GIAN TRUY VẤN ---
start_time = datetime.strptime('2018-06-11 11:30:00', '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)     # Thay đổi thời gian sàn 
end_time = datetime.now(timezone.utc)

# --- BATCH CONFIG ---
BATCH_SIZE_MINUTES = 30000  # tương ứng 30,000 dòng

# --- TẠO TÊN FILE TỪ end_time ---
FILE_PREFIX = "TRX"                                 # Biến tiền tố có thể thay đổi
FILE_NAME = end_time.strftime('%Y-%m-%d_%H-%M-%S')  # Định dạng để tránh ký tự không hợp lệ trong tên file
OUTPUT_FILE = f'data/{FILE_PREFIX}/{FILE_PREFIX}_{FILE_NAME}.csv'          

# --- KIỂM TRA THƯ MỤC data VÀ LƯU FILE ---
os.makedirs(f'data/{FILE_PREFIX}', exist_ok=True)              
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# --- TẠO CLIENT MỚI CHO MỖI THREAD ---
def create_client():
    return get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=DATABASE_NAME
    )

# --- LẤY DANH SÁCH CỘT ---
def get_column_list(table_name):
    client = create_client()
    query = f"""
        SELECT name 
        FROM system.columns 
        WHERE database = '{DATABASE_NAME}' 
          AND table = '{table_name}'
    """
    result = client.query(query)
    return [row['name'] for row in result.named_results()]

# --- TRUY VẤN 1 BATCH ---
def fetch_batch_data(table_name, columns, start, end):
    client = create_client()
    start_str = start.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end.strftime('%Y-%m-%d %H:%M:%S')

    # Thêm điều kiện 'Valid = 1' nếu là bảng A
    extra_condition = "AND Valid = '1'" if table_name == TABLE_A and 'Valid' in columns else ""

    query = f"""
        SELECT {', '.join(columns)}
        FROM {table_name} FINAL
        WHERE OpenTime >= toDateTime('{start_str}')
          AND OpenTime < toDateTime('{end_str}')
          {extra_condition}
        ORDER BY OpenTime
    """

    result = client.query(query)
    data = result.named_results()
    df = pd.DataFrame(data)

    if 'OpenTime' in df.columns:
        df['OpenTime'] = pd.to_datetime(df['OpenTime']).dt.floor('s') # chuẩn hóa giây
        df = df.drop_duplicates(subset=['OpenTime']) # loại bỏ trùng theo OpenTime

    return df

# --- XỬ LÝ TOÀN BỘ ---
def process_all_batches():
    # Lấy danh sách cột đầy đủ từ bảng A và B
    all_columns_a = get_column_list(TABLE_A)
    all_columns_b = get_column_list(TABLE_B)

    # Ưu tiên giữ cột từ bảng A nếu trùng, ngoại trừ OpenTime vẫn giữ lại để merge
    columns_b_unique = [col for col in all_columns_b if col not in all_columns_a or col == 'OpenTime']

    # Tính tổng số batch dựa vào khoảng thời gian
    current_start = start_time
    total_minutes = int((end_time - start_time).total_seconds() / 60)
    total_batches = (total_minutes + BATCH_SIZE_MINUTES - 1) // BATCH_SIZE_MINUTES

    for i in tqdm(range(total_batches), desc="Đang tải dữ liệu"):
        current_end = current_start + timedelta(minutes=BATCH_SIZE_MINUTES)

        try:
            # Truy vấn song song cả A và B
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(fetch_batch_data, TABLE_A, all_columns_a, current_start, current_end)
                future_b = executor.submit(fetch_batch_data, TABLE_B, columns_b_unique, current_start, current_end)

                df_a = future_a.result()
                df_b = future_b.result()

            # Gộp hai bảng theo OpenTime
            merged = pd.merge(df_a, df_b, on='OpenTime', how='inner')
            merged = merged.sort_values('OpenTime')

            # Ghi ra file
            merged.to_csv(OUTPUT_FILE, mode='a', index=False, header=(i == 0))

        except Exception as e:
            print(f"\nLỗi ở batch {i + 1}: {e}\n")

        # Cập nhật thời gian bắt đầu cho batch tiếp theo    
        current_start = current_end

        # In thời gian cuối cùng của batch hiện tại (debug)
        if not df_a.empty:
            print("\nLast A:", df_a['OpenTime'].max())
        if not df_b.empty:
            print("Last B:", df_b['OpenTime'].max())

# --- CHẠY ---
if __name__ == "__main__":
    process_all_batches()
    print(f"\nDữ liệu đã được lưu hoàn chỉnh vào: {OUTPUT_FILE}")
