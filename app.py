"""
Ứng dụng Streamlit: dự đoán gian lận thủ công + phân tích độ quan trọng đặc trưng.
Chạy:  streamlit run app.py
Cần có các file do train_stacking.py tạo ra trong cùng thư mục.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from features import ALL_FEATURES, add_features

st.set_page_config(page_title="Phát hiện gian lận (Stacking)", page_icon="🔍", layout="wide")
st.title("🔍 Hệ thống phát hiện gian lận – Stacking Ensemble")

REQUIRED = ["fraud_stacking.joblib", "metrics.json",
            "feature_importance.csv", "permutation_importance.csv"]
missing = [f for f in REQUIRED if not Path(f).exists()]
if missing:
    st.error(f"Thiếu file: {', '.join(missing)}. Hãy chạy `python train_stacking.py` trước.")
    st.stop()


@st.cache_resource
def load_artifacts():
    bundle = joblib.load("fraud_stacking.joblib")
    metrics = json.load(open("metrics.json", encoding="utf-8"))
    imp = pd.read_csv("feature_importance.csv")
    perm = pd.read_csv("permutation_importance.csv")
    return bundle["model"], bundle["threshold"], metrics, imp, perm


model, threshold, metrics, imp_df, perm_df = load_artifacts()

tab_pred, tab_perf, tab_imp = st.tabs(
    ["🧮 Dự đoán thủ công", "📈 Hiệu năng mô hình", "📊 Độ quan trọng & Giữ/Bỏ đặc trưng"])

# ============================================================ TAB 1: DỰ ĐOÁN
with tab_pred:
    st.subheader("Nhập thông tin giao dịch")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tx_type = st.selectbox("Loại giao dịch (type)",
                                   ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])
            step = st.number_input("Step (giờ thứ mấy, 1–743)", min_value=1, max_value=743,
                                   value=1, step=1)
            amount = st.number_input("Số tiền giao dịch (amount)", min_value=0.0,
                                     value=181000.0, step=1000.0, format="%.2f")
        with c2:
            old_org = st.number_input("Số dư người gửi TRƯỚC (oldbalanceOrg)", min_value=0.0,
                                      value=181000.0, step=1000.0, format="%.2f")
            new_org = st.number_input("Số dư người gửi SAU (newbalanceOrig)", min_value=0.0,
                                      value=0.0, step=1000.0, format="%.2f")
        with c3:
            old_dst = st.number_input("Số dư người nhận TRƯỚC (oldbalanceDest)", min_value=0.0,
                                      value=0.0, step=1000.0, format="%.2f")
            new_dst = st.number_input("Số dư người nhận SAU (newbalanceDest)", min_value=0.0,
                                      value=0.0, step=1000.0, format="%.2f")
        submitted = st.form_submit_button("🚀 Dự đoán", use_container_width=True)

    if submitted:
        raw = pd.DataFrame([{
            "step": step, "type": tx_type, "amount": amount,
            "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
            "oldbalanceDest": old_dst, "newbalanceDest": new_dst,
        }])
        X_in = add_features(raw)[ALL_FEATURES]
        proba = float(model.predict_proba(X_in)[0, 1])
        is_fraud = proba >= threshold

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Xác suất gian lận", f"{proba:.2%}")
        m2.metric("Ngưỡng quyết định", f"{threshold:.3f}")
        m3.metric("Kết luận", "GIAN LẬN" if is_fraud else "BÌNH THƯỜNG")
        st.progress(min(proba, 1.0))
        if is_fraud:
            st.error("⚠️ Giao dịch bị đánh dấu NGHI GIAN LẬN – nên kiểm tra thủ công.")
        else:
            st.success("✅ Giao dịch được đánh giá BÌNH THƯỜNG.")

        with st.expander("Đặc trưng tự động tính thêm"):
            st.dataframe(X_in[["errorBalanceOrig", "errorBalanceDest"]], hide_index=True)
            st.caption("errorBalance = chênh lệch giữa số dư thực tế và số dư lý thuyết sau giao dịch. "
                       "Khác 0 nghĩa là số dư không khớp với số tiền chuyển.")

# ============================================================ TAB 2: HIỆU NĂNG
with tab_perf:
    st.subheader("Chỉ số trên tập test (phân phối thật, chưa cân bằng)")
    keys = ["Precision", "Recall", "F1-Score", "AUC-ROC", "PR-AUC"]
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        col.metric(k, f"{metrics[k]:.3f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bình thường", "Gian lận"],
                yticklabels=["Bình thường", "Gian lận"], ax=ax)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    st.pyplot(fig)

# ============================================================ TAB 3: ĐẶC TRƯNG
with tab_imp:
    st.subheader("Độ quan trọng của đặc trưng")
    colors = imp_df["Decision"].map({"Giữ": "#2e9e5b", "Loại bỏ": "#d64545"})

    left, right = st.columns(2)
    with left:
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        ax1.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color=colors[::-1])
        ax1.set_xlabel("Độ quan trọng (trung bình RF + LightGBM + XGBoost)")
        ax1.set_title("Xanh = nên giữ · Đỏ = có thể loại bỏ")
        st.pyplot(fig1)
    with right:
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        ax2.plot(range(1, len(imp_df) + 1), imp_df["Cumulative"], marker="o",
                 color="crimson", linewidth=2)
        ax2.axhline(0.95, color="green", linestyle="--", label="Ngưỡng 95%")
        ax2.set_xticks(range(1, len(imp_df) + 1))
        ax2.set_xlabel("Số lượng đặc trưng")
        ax2.set_ylabel("Độ quan trọng tích lũy")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    st.subheader("Permutation importance của toàn bộ mô hình stacking")
    st.caption("Xáo trộn từng đặc trưng gốc rồi đo PR-AUC giảm bao nhiêu. "
               "Giá trị ≈ 0 hoặc âm nghĩa là mô hình gần như không dùng đặc trưng đó.")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    p = perm_df.sort_values("Mean")
    ax3.barh(p["Feature"], p["Mean"], xerr=p["Std"], color="#3b78c4")
    ax3.set_xlabel("Mức giảm PR-AUC khi xáo trộn")
    st.pyplot(fig3)

    st.subheader("💡 Khuyến nghị")
    keep = imp_df[imp_df["Decision"] == "Giữ"]
    drop = imp_df[imp_df["Decision"] == "Loại bỏ"]
    k_col, d_col = st.columns(2)
    with k_col:
        st.success("✅ NÊN GIỮ LẠI")
        for _, r in keep.iterrows():
            st.write(f"- **{r['Feature']}** ({r['Importance']:.3f})")
    with d_col:
        st.error("❌ CÓ THỂ LOẠI BỎ")
        if drop.empty:
            st.write("Không có đặc trưng nào thừa.")
        for _, r in drop.iterrows():
            st.write(f"- **{r['Feature']}** ({r['Importance']:.3f})")
    st.info("Sau khi loại bỏ, hãy huấn luyện lại và so sánh F1/PR-AUC để chắc chắn "
            "hiệu năng không giảm trước khi áp dụng thật.")
