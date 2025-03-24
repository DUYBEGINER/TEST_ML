import datetime
import os
import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Danh sách IP
ip_list = [
    "185.190.58.108", "192.168.1.100", "212.129.2.219", "5.188.10.176",
    "172.16.254.1", "203.0.113.5", "198.51.100.10", "10.10.10.10",
    "45.67.89.123", "78.90.12.34", "123.45.67.89", "94.63.177.105",
    "221.194.47.239", "125.120.183.224", "119.193.140.213", "45.67.89.555"
] + [f"10.0.0.{random.randint(1, 255)}" for _ in range(20)]

# Hàm tạo dataset train
def generate_train_dataset(num_samples=100000):
    data = {
        "ip": [],
        "failed_attempts": [],
        "attempt_freq": [],
        "min_time_between_attempts": [],
        "num_unique_usernames": []
    }

    for _ in range(num_samples):
        ip = random.choice(ip_list)
        failed_attempts = random.randint(0, 20)
        total_attempts = failed_attempts
        attempt_freq = total_attempts / 5.0 if total_attempts > 0 else 0
        min_time_between = random.uniform(0, 15) if total_attempts > 1 else 0
        num_unique_usernames = random.randint(1, min(10, failed_attempts + 1)) if failed_attempts > 0 else 1

        data["ip"].append(ip)
        data["failed_attempts"].append(failed_attempts)
        data["attempt_freq"].append(attempt_freq)
        data["min_time_between_attempts"].append(min_time_between)
        data["num_unique_usernames"].append(num_unique_usernames)

    df = pd.DataFrame(data)

    # Gán nhãn: Cần ít nhất 2 điều kiện đúng
    conditions = [
        df["failed_attempts"] >= 10,
        df["attempt_freq"] > 5.0,
        df["min_time_between_attempts"] < 5.0,
        df["num_unique_usernames"] >= 5
    ]
    df["label"] = (sum(conditions) >= 2).astype(int)

    return df

# Hàm tạo dataset test
def generate_test_dataset(num_samples=50000):  # Sửa thành 50,000 mẫu
    data = {
        "ip": [],
        "failed_attempts": [],
        "attempt_freq": [],
        "min_time_between_attempts": [],
        "num_unique_usernames": []
    }

    for _ in range(num_samples):
        ip = random.choice(ip_list)
        failed_attempts = random.randint(0, 300)
        total_attempts = failed_attempts
        attempt_freq = total_attempts / 5.0 if total_attempts > 0 else 0
        min_time_between = random.uniform(0, 200) if total_attempts > 1 else 0
        num_unique_usernames = random.randint(1, 200) if total_attempts > 0 else 1

        data["ip"].append(ip)
        data["failed_attempts"].append(failed_attempts)
        data["attempt_freq"].append(attempt_freq)
        data["min_time_between_attempts"].append(min_time_between)
        data["num_unique_usernames"].append(num_unique_usernames)

    df = pd.DataFrame(data)

    # Gán nhãn: Cần ít nhất 2 điều kiện đúng
    conditions = [
        df["failed_attempts"] >= 10,
        df["attempt_freq"] > 5.0,
        df["min_time_between_attempts"] < 5.0,
        df["num_unique_usernames"] >= 5
    ]
    df["label"] = (sum(conditions) >= 2).astype(int)

    return df

# Hàm huấn luyện và đánh giá mô hình (bỏ SMOTE)
def train_and_evaluate(train_file="train_features_with_labels_100k.csv", test_file="train_features_with_labels_test_50k.csv"):
    # Đọc dataset train
    df_train = pd.read_csv(train_file)
    X_train = df_train[["failed_attempts", "attempt_freq",
                        "min_time_between_attempts", "num_unique_usernames"]].values
    y_train = df_train["label"].values

    # Đọc dataset test
    df_test = pd.read_csv(test_file)
    X_test = df_test[["failed_attempts", "attempt_freq",
                      "min_time_between_attempts", "num_unique_usernames"]].values
    y_test = df_test["label"].values

    # Thống kê tỉ lệ nhãn trong tập train
    label_counts_train = pd.Series(y_train).value_counts()
    total_samples_train = len(y_train)
    label_0_ratio_train = label_counts_train.get(0, 0) / total_samples_train * 100
    label_1_ratio_train = label_counts_train.get(1, 0) / total_samples_train * 100

    print("\nThống kê tỉ lệ theo nhãn trong tập train:")
    print(f"Nhãn 0 (Không phải brute force): {label_counts_train.get(0, 0)} mẫu - {label_0_ratio_train:.2f}%")
    print(f"Nhãn 1 (Brute force): {label_counts_train.get(1, 0)} mẫu - {label_1_ratio_train:.2f}%")

    # Vẽ biểu đồ tròn minh họa tỉ lệ nhãn trong tập train
    plt.figure(figsize=(6, 6))
    plt.pie([label_counts_train.get(0, 0), label_counts_train.get(1, 0)],
            labels=["Not Brute Force (0)", "Brute Force (1)"],
            autopct="%1.2f%%", colors=["skyblue", "salmon"], startangle=90)
    plt.title("Tỉ lệ nhãn trong tập train")
    plt.savefig("label_ratio_pie_chart_train.png")
    plt.close()
    print("Đã lưu biểu đồ tỉ lệ nhãn tập train vào 'label_ratio_pie_chart_train.png'")

    # Thống kê tỉ lệ nhãn trong tập test
    label_counts_test = pd.Series(y_test).value_counts()
    total_samples_test = len(y_test)
    label_0_ratio_test = label_counts_test.get(0, 0) / total_samples_test * 100
    label_1_ratio_test = label_counts_test.get(1, 0) / total_samples_test * 100

    print("\nThống kê tỉ lệ theo nhãn trong tập test:")
    print(f"Nhãn 0 (Không phải brute force): {label_counts_test.get(0, 0)} mẫu - {label_0_ratio_test:.2f}%")
    print(f"Nhãn 1 (Brute force): {label_counts_test.get(1, 0)} mẫu - {label_1_ratio_test:.2f}%")

    # Vẽ biểu đồ tròn minh họa tỉ lệ nhãn trong tập test
    plt.figure(figsize=(6, 6))
    plt.pie([label_counts_test.get(0, 0), label_counts_test.get(1, 0)],
            labels=["Not Brute Force (0)", "Brute Force (1)"],
            autopct="%1.2f%%", colors=["skyblue", "salmon"], startangle=90)
    plt.title("Tỉ lệ nhãn trong tập test")
    plt.savefig("label_ratio_pie_chart_test.png")
    plt.close()
    print("Đã lưu biểu đồ tỉ lệ nhãn tập test vào 'label_ratio_pie_chart_test.png'")

    # Huấn luyện mô hình trên tập train (không dùng SMOTE)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "bruteforce_detector.pkl")
    print(f"Đã huấn luyện mô hình trên tập train từ {train_file} và lưu vào bruteforce_detector.pkl")

    # Đánh giá trên tập train
    y_train_pred = model.predict(X_train)
    print("\nĐánh giá trên tập huấn luyện:")
    print("Classification Report:")
    print(classification_report(y_train, y_train_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")

    # Đánh giá trên tập test
    y_test_pred = model.predict(X_test)
    print("\nĐánh giá trên tập kiểm tra:")
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    # Vẽ biểu đồ ma trận nhầm lẫn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Brute Force", "Brute Force"],
                yticklabels=["Not Brute Force", "Brute Force"])
    plt.title("Confusion Matrix on Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix_test.png")
    plt.close()
    print("Đã lưu biểu đồ ma trận nhầm lẫn vào 'confusion_matrix_test.png'")

    return model

# Test với log từ ssh_log_fake.txt
def test_model(test_log_file="ssh_log_fake.txt"):
    if not os.path.exists(test_log_file):
        print(f"File {test_log_file} không tồn tại.")
        return

    # Load mô hình
    model = joblib.load("bruteforce_detector.pkl")

    # Parse log từ file txt
    def parse_ssh_log(log_lines):
        data = []
        current_year = 2024
        for line in log_lines:
            parts = line.split()
            timestamp_str = " ".join(parts[0:3])
            timestamp = datetime.strptime(f"{timestamp_str} {current_year}", "%b %d %H:%M:%S %Y")
            ip = parts[-4]
            username_start = line.find("for") + 4
            username_end = line.find("from")
            username = line[username_start:username_end].replace("invalid user", "").strip()

            if "Accepted" in line:
                success = True
            elif "Failed" in line:
                success = False
            else:
                continue
            data.append([timestamp, ip, username, success])
        return pd.DataFrame(data, columns=["timestamp", "ip", "username", "success"])

    def extract_features_extended(log_df, time_window=datetime.timedelta(minutes=5)):
        features = []
        unique_ips = log_df["ip"].unique()

        for ip in unique_ips:
            ip_data = log_df[log_df["ip"] == ip].sort_values("timestamp")
            recent = ip_data[ip_data["timestamp"] > (ip_data["timestamp"].max() - time_window)]

            failed_attempts = len(recent[recent["success"] == False])
            total_attempts = failed_attempts  # Chỉ tính failed_attempts
            attempt_freq = total_attempts / (time_window.total_seconds() / 60)

            if len(ip_data) > 1:
                time_diffs = ip_data["timestamp"].diff().dt.total_seconds().dropna()
                min_time_between = time_diffs.min()
            else:
                min_time_between = 0

            num_unique_usernames = len(ip_data["username"].unique())

            features.append([ip, failed_attempts, attempt_freq,
                             min_time_between, num_unique_usernames])

        return pd.DataFrame(features, columns=["ip", "failed_attempts", "attempt_freq",
                                               "min_time_between_attempts", "num_unique_usernames"])

    # Đọc và xử lý log test
    with open(test_log_file, "r") as f:
        test_log_lines = f.readlines()

    test_log_df = parse_ssh_log(test_log_lines)
    test_features_df = extract_features_extended(test_log_df)

    # Dự đoán
    X_test = test_features_df[["failed_attempts", "attempt_freq",
                               "min_time_between_attempts", "num_unique_usernames"]].values
    predictions = model.predict(X_test)

    # Thêm cột dự đoán
    test_features_df["predicted_label"] = predictions
    print("\nKết quả dự đoán trên dữ liệu test từ ssh_log_fake.txt:")
    print(test_features_df)

    # Lưu kết quả test
    test_features_df.to_csv("test_features_with_predictions.csv", index=False)
    print("Đã lưu kết quả test vào test_features_with_predictions.csv")

# Chạy chương trình
if __name__ == "__main__":
    # Tạo dataset train
    print("Generating train dataset...")
    train_dataset = generate_train_dataset(1000000)
    train_dataset.to_csv("train_features_with_labels_100k.csv", index=False)
    print("Đã tạo dataset huấn luyện với 100,000 dòng và lưu vào train_features_with_labels_100k.csv")
    print(train_dataset.head())

    # Tạo dataset test
    print("\nGenerating test dataset...")
    test_dataset = generate_test_dataset(500000)
    test_dataset.to_csv("train_features_with_labels_test_50k.csv", index=False)
    print("Đã tạo dataset kiểm tra với 50,000 dòng và lưu vào train_features_with_labels_test_50k.csv")
    print(test_dataset.head())

    # Huấn luyện và đánh giá mô hình
    train_and_evaluate("train_features_with_labels_100k.csv", "train_features_with_labels_test_50k.csv")

    # Test với ssh_log_fake.txt (tùy chọn)
    # test_model("ssh_log_fake.txt")