import datetime
import os
import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
data = pd.read_csv("synthetic_bruteforce_data_with_ip.csv")
data = data.drop(columns=['ip_address'])

# Chia dữ liệu thành đặc trưng (X) và nhãn (y)
X = data.drop(columns=['attack_detected'])
y = data['attack_detected']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

# Chuẩn hóa dữ liệu (chỉ áp dụng cho các cột số)
scaler = StandardScaler()
numerical_cols = ['login_attempts', 'ip_reputation_score', 'failed_logins', 'num_unique_usernames', 'min_time_between_attempts']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = rf_model.predict(X_test)

# Đánh giá mô hìnhgi
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Đánh giá mô hình:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=['No Attack', 'Attack']))

# Vẽ confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Attack', 'Attack'], yticklabels=['No Attack', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Lưu mô hình và scaler
joblib.dump(rf_model, 'bruteforce_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Mô hình đã được lưu vào 'bruteforce_rf_model.pkl'")
print("Scaler đã được lưu vào 'scaler.pkl'")

# Tầm quan trọng của các đặc trưng
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTầm quan trọng của các đặc trưng:")
print(feature_importance)

# Vẽ biểu đồ tầm quan trọng của đặc trưng
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Tính toán giá trị xác suất dự đoán
y_probs = rf_model.predict_proba(X_test)[:, 1]  # Chỉ lấy xác suất cho lớp 'Attack'

# Tính toán FPR, TPR và ngưỡng
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Vẽ đường cong ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Đường chéo tham chiếu
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print(f"AUC-ROC Score: {roc_auc:.4f}")