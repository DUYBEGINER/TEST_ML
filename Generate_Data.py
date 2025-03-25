import pandas as pd
import numpy as np

def generate_random_ip():
    """
    Hàm tạo địa chỉ IP ngẫu nhiên dưới dạng chuỗi (ví dụ: '192.168.1.100').

    Returns:
    - ip (str): Địa chỉ IP ngẫu nhiên.
    """
    return '.'.join(str(np.random.randint(0, 256)) for _ in range(4))

def generate_synthetic_data(n_samples):
    """
    Hàm tạo dữ liệu ngẫu nhiên để phát hiện tấn công brute-force, bao gồm cột ip_address.

    Parameters:
    - n_samples (int): Số lượng mẫu dữ liệu cần tạo.

    Returns:
    - synthetic_data (DataFrame): Dataset với các cột đã chọn, ip_address và nhãn attack_detected.
    """
    # Khởi tạo dữ liệu ngẫu nhiên
    synthetic_data = pd.DataFrame({
        'ip_address': [generate_random_ip() for _ in range(n_samples)],
        'login_attempts': np.random.randint(1, 51, size=n_samples),  # Từ 1 đến 50
        'ip_reputation_score': np.random.uniform(0, 1, size=n_samples),  # Từ 0 đến 1
    })

    # Tạo num_unique_usernames dựa trên login_attempts
    synthetic_data['num_unique_usernames'] = synthetic_data['login_attempts'].apply(
        lambda x: np.random.randint(1, min(x + 1, 21))  # Từ 1 đến min(login_attempts, 20)
    )

    # Tạo failed_logins (phải nhỏ hơn hoặc bằng login_attempts)
    synthetic_data['failed_logins'] = synthetic_data.apply(
        lambda row: np.random.randint(0, row['login_attempts'] + 1), axis=1
    )

    # Tạo min_time_between_attempts (thời gian tối thiểu giữa các lần thử, tính bằng giây)
    # Nếu login_attempts = 1, đặt giá trị mặc định là 0 (vì không có khoảng thời gian giữa các lần thử)
    synthetic_data['min_time_between_attempts'] = synthetic_data['login_attempts'].apply(
        lambda x: 0 if x == 1 else np.min(np.random.uniform(0.01, 10, size=x-1))
    )

    # Giới hạn giá trị (clipping) để đảm bảo hợp lý
    synthetic_data['login_attempts'] = synthetic_data['login_attempts'].clip(lower=1, upper=50)
    synthetic_data['failed_logins'] = synthetic_data['failed_logins'].clip(
        lower=0, upper=synthetic_data['login_attempts']
    )
    synthetic_data['num_unique_usernames'] = synthetic_data['num_unique_usernames'].clip(lower=1, upper=20)
    synthetic_data['min_time_between_attempts'] = synthetic_data['min_time_between_attempts'].clip(lower=0, upper=10)
    synthetic_data['ip_reputation_score'] = synthetic_data['ip_reputation_score'].clip(lower=0, upper=1)

    # Tính từng điều kiện (1 nếu đúng, 0 nếu sai)
    condition1 = (synthetic_data['failed_logins'] >= 20).astype(int)  # Ngưỡng cho failed_logins
    condition2 = (synthetic_data['login_attempts'] >= 20).astype(int)  # Ngưỡng cho login_attempts
    condition3 = (synthetic_data['ip_reputation_score'] < 0.5).astype(int)  # Ngưỡng cho ip_reputation_score
    condition4 = (synthetic_data['num_unique_usernames'] >= 5).astype(int)  # Ngưỡng cho num_unique_usernames
    condition5 = (synthetic_data['min_time_between_attempts'] <= 0.5).astype(int)  # Ngưỡng cho min_time_between_attempts

    # Gán trọng số cho các điều kiện
    weights = {
        'condition1': 0.2,  # failed_logins
        'condition2': 0.2,  # login_attempts
        'condition3': 0.05,  # ip_reputation_score
        'condition4': 0.35,  # num_unique_usernames
        'condition5': 0.2   # min_time_between_attempts
    }

    # Tính tổng trọng số của các điều kiện thỏa mãn
    weighted_sum = (
        condition1 * weights['condition1'] +
        condition2 * weights['condition2'] +
        condition3 * weights['condition3'] +
        condition4 * weights['condition4'] +
        condition5 * weights['condition5']
    )

    # Gán nhãn: 1 nếu tổng trọng số >= 0.5
    synthetic_data['attack_detected'] = (weighted_sum >= 0.5).astype(int)

    # Điều chỉnh tỷ lệ attack_detected (khoảng 45%)
    target_ratio = 0.45
    current_ratio = synthetic_data['attack_detected'].mean()

    if current_ratio > target_ratio:
        # Cập nhật mask để phù hợp với ngưỡng mới
        mask = (synthetic_data['attack_detected'] == 1) & (
            (synthetic_data['failed_logins'].between(15, 20)) |  # Gần ngưỡng
            (synthetic_data['login_attempts'].between(18, 20)) |  # Gần ngưỡng
            (synthetic_data['ip_reputation_score'].between(0.48, 0.5)) |  # Gần ngưỡng
            (synthetic_data['num_unique_usernames'].between(8, 10)) |  # Gần ngưỡng
            (synthetic_data['min_time_between_attempts'].between(0.4, 0.5))  # Gần ngưỡng
        )
        n_to_flip = min(int((current_ratio - target_ratio) * n_samples), mask.sum())
        if n_to_flip > 0:
            flip_indices = synthetic_data[mask].sample(n_to_flip).index
            synthetic_data.loc[flip_indices, 'attack_detected'] = 0

    # Sắp xếp lại các cột theo đúng thứ tự (dựa trên yêu cầu trước đó)
    synthetic_data = synthetic_data[[
        'ip_address', 'login_attempts', 'ip_reputation_score', 'num_unique_usernames',
        'failed_logins', 'min_time_between_attempts', 'attack_detected'
    ]]

    return synthetic_data

# Tạo 1,000,000 mẫu dữ liệu
n_samples = 1000000
synthetic_data = generate_synthetic_data(n_samples)

# Kiểm tra dataset
print(synthetic_data.head())
print("\nTỷ lệ attack_detected:", synthetic_data['attack_detected'].mean())

# Lưu dataset
synthetic_data.to_csv('synthetic_bruteforce_data_with_ip.csv', index=False)

# Đọc dataset và thống kê tỷ lệ nhãn
df = pd.read_csv('synthetic_bruteforce_data_with_ip.csv')

y = df["attack_detected"].values
label_counts_train = pd.Series(y).value_counts()
total_samples_train = len(y)
label_0_ratio_train = label_counts_train.get(0, 0) / total_samples_train * 100
label_1_ratio_train = label_counts_train.get(1, 0) / total_samples_train * 100

print("\nThống kê tỉ lệ theo nhãn trong tập train:")
print(f"Nhãn 0 (Không phải brute force): {label_counts_train.get(0, 0)} mẫu - {label_0_ratio_train:.2f}%")
print(f"Nhãn 1 (Brute force): {label_counts_train.get(1, 0)} mẫu - {label_1_ratio_train:.2f}%")
print("Dataset đã được lưu vào 'synthetic_bruteforce_data_with_ip.csv'")