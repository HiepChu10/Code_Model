# ================================ READ ME! ================================
#           Thay đổi đường dẫn và tên thư mục trước khi sử dụng


# Query từ sàn về để check
import requests
from datetime import datetime, timezone

def to_milliseconds(dt_str):
    """Chuyển chuỗi thời gian UTC thành timestamp (epoch milliseconds)"""
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)  # Gán múi giờ UTC chính xác
    return int(dt.timestamp() * 1000)

# Cấu hình truy vấn (giờ UTC)
symbol = "TRXUSDT"
interval = "1m"
start_str = "2018-06-26 01:55:00"  # UTC
end_str = "2018-06-26 12:10:00"    # UTC

# Chuyển sang epoch milliseconds UTC
start_time = to_milliseconds(start_str)
end_time = to_milliseconds(end_str)

# Gọi API Binance
url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": symbol,
    "interval": interval,
    "startTime": start_time,
    "endTime": end_time,
    "limit": 1000
}

# Gửi request
response = requests.get(url, params=params)

# Xử lý dữ liệu
if response.status_code == 200:
    data = response.json()
    print(f"Lấy được {len(data)} cây nến:")
    for kline in data:
        open_time = datetime.utcfromtimestamp(kline[0] / 1000)
        print(f"{open_time} | Open: {kline[1]} High: {kline[2]} Low: {kline[3]} Close: {kline[4]} Volume: {kline[5]}")
else:
    print(f"Lỗi truy vấn: {response.status_code} - {response.text}")

print("=======================================================================")
print("Start Time (UTC ms):", start_time)
print("End Time (UTC ms):  ", end_time)