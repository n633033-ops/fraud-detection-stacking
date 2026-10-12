# Phát hiện gian lận giao dịch bằng Stacking Ensemble
## 📊 Dữ liệu sử dụng (Dataset)

Dự án này sử dụng bộ dữ liệu **PaySim** mô phỏng các giao dịch tài chính trên thiết bị di động để phát hiện gian lận. 
Do dung lượng file khá lớn, dữ liệu không được tải trực tiếp lên kho lưu trữ này. Bạn có thể xem chi tiết và tải file dữ liệu gốc (`.csv`) tại Kaggle:

👉 **[Synthetic Financial Datasets For Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1)**

**Thông tin sơ lược về dữ liệu:**
*   Dữ liệu bao gồm các đặc trưng như: `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`,...
*   Biến mục tiêu (Target variable): `isFraud` (1 là gian lận, 0 là bình thường).
Dự án xây dựng mô hình phát hiện giao dịch tài chính gian lận trên dữ liệu mất cân bằng nghiêm trọng, kết hợp ba mô hình (Random Forest, LightGBM, XGBoost) bằng kỹ thuật **stacking**, kèm ứng dụng **Streamlit** để nhập giao dịch thủ công và xem phân tích độ quan trọng của đặc trưng.

> Dự án được thực hiện với sự hỗ trợ của công cụ AI (Claude) trong quá trình viết code; phần chạy thực nghiệm, kiểm tra kết quả và phân tích do mình thực hiện.

## 1. Bài toán

Trong dữ liệu giao dịch, gian lận chỉ chiếm một phần rất nhỏ (khoảng 0,1%), nên một mô hình đoán "mọi giao dịch đều bình thường" vẫn đạt accuracy rất cao nhưng vô dụng. Mục tiêu là bắt được nhiều vụ gian lận nhất có thể (Recall cao) mà không cảnh báo nhầm quá nhiều (Precision cao).

## 2. Dữ liệu

- Nguồn: **PaySim – Synthetic Financial Datasets For Fraud Detection** (Kaggle), link: *[dán link Kaggle vào đây]*
- Đây là dữ liệu **giả lập** mô phỏng giao dịch tiền di động, khoảng 6,3 triệu dòng.
- Số dòng đã dùng để huấn luyện: *[điền số dòng bạn đã dùng, ví dụ toàn bộ hoặc 2.000.000 dòng đầu]*
- Dữ liệu không được đưa lên repo vì dung lượng lớn (~470 MB).

## 3. Phương pháp

- **Đặc trưng:** giữ các cột `type`, `step`, `amount` và 4 cột số dư; tạo thêm hai đặc trưng chênh lệch số dư `errorBalanceOrig` và `errorBalanceDest` (số dư thực tế lệch bao nhiêu so với số dư lý thuyết sau giao dịch).
- **Chia dữ liệu:** 80% huấn luyện, 20% kiểm tra (chia ngẫu nhiên có phân tầng theo nhãn). Từ phần huấn luyện tách thêm tập validation để chọn ngưỡng.
- **Cân bằng lớp:** undersample lớp đa số (tối đa 20 giao dịch bình thường cho mỗi giao dịch gian lận) **chỉ trên tập huấn luyện**. Không dùng SMOTE vì stacking dùng cross-validation nội bộ, SMOTE chạy trước sẽ làm mẫu tổng hợp rò rỉ giữa các fold.
- **Mô hình stacking:** Random Forest + LightGBM + XGBoost làm mô hình nền, Logistic Regression làm mô hình tổng hợp, cross-validation 5 fold.
- **Ngưỡng quyết định:** chọn ngưỡng tối ưu F1 trên tập validation (không dùng mặc định 0,5).
- **Độ quan trọng đặc trưng:** trung bình độ quan trọng của ba mô hình nền (kèm biểu đồ tích lũy, ngưỡng 95%) và permutation importance trên toàn bộ mô hình stacking.

## 4. Kết quả

Đánh giá trên tập test gồm 400.000 giao dịch (407 vụ gian lận), giữ nguyên phân phối thật:

| Chỉ số | Giá trị |
|---|---|
| Precision | 1,000 |
| Recall | 0,983 |
| F1-Score | 0,991 |
| AUC-ROC | 0,998 |
| PR-AUC | 0,988 |

Ma trận nhầm lẫn:

| | Dự đoán: Bình thường | Dự đoán: Gian lận |
|---|---|---|
| **Thực tế: Bình thường** | 399.593 | 0 |
| **Thực tế: Gian lận** | 7 | 400 |

Mô hình bắt đúng 400/407 vụ gian lận và không cảnh báo nhầm giao dịch bình thường nào.

## 5. Độ quan trọng của đặc trưng

Permutation importance cho thấy mô hình gần như chỉ dựa vào hai đặc trưng liên quan đến số dư người gửi là `errorBalanceOrig` và `newbalanceOrig`; các đặc trưng còn lại đóng góp gần bằng 0 khi đo bằng cách xáo trộn. Trong khi đó, độ quan trọng trung bình theo cây phân bổ đều hơn cho nhiều đặc trưng, nên hai cách đo cho khuyến nghị giữ/loại bỏ khác nhau. Cần huấn luyện lại chỉ với các đặc trưng quan trọng và so sánh hiệu năng trước khi loại bỏ chúng.

## 6. Hạn chế

- Dữ liệu **giả lập**, kết quả gần hoàn hảo khó lặp lại trên dữ liệu thực tế.
- Các cột số dư (`newbalance*`) có thể gây **rò rỉ thông tin** (ví dụ giao dịch rút sạch tiền rồi số dư về 0), làm mô hình phụ thuộc quá mạnh vào chúng.
- Tập test chỉ có 407 vụ gian lận nên các chỉ số gần 1,0 dựa trên số ca dương khá ít.
- Chia train/test ngẫu nhiên, chưa chia theo thời gian (cột `step`) như tình huống triển khai thực tế.
- Chưa so sánh stacking với từng mô hình đơn lẻ để định lượng mức cải thiện.
- Ngưỡng chọn theo F1, chưa tối ưu theo chi phí thực tế của việc bỏ sót và cảnh báo nhầm.

## 7. Cấu trúc dự án

```
features.py         hàm tạo đặc trưng, dùng chung cho huấn luyện và ứng dụng
train_stacking.py   huấn luyện mô hình, chọn ngưỡng, tính độ quan trọng, lưu kết quả
app.py              ứng dụng Streamlit (dự đoán thủ công, hiệu năng, đặc trưng)
requirements.txt    danh sách thư viện
```

## 8. Cách chạy

```bash
# 1. Tải dataset PaySim từ Kaggle, đổi tên file thành Fraud.csv và đặt cùng thư mục với code
# 2. Cài thư viện
pip install -r requirements.txt

# 3. Huấn luyện (sinh ra fraud_stacking.joblib, metrics.json, các file CSV)
python train_stacking.py

# 4. Mở ứng dụng
streamlit run app.py
```

