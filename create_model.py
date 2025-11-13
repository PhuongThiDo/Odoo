import streamlit as st
import pandas as pd
import pickle
import xmlrpc.client

# --- Cấu hình Odoo ---
# Thay URL bằng ngrok public URL của bạn
url = "https://brenton-chevronny-kristi.ngrok-free.dev"
db = "Odoo"          # database name trên Odoo
username = "23070615@vnu.edu.vn"      # user Odoo
password = "Dothiphuong99"   # mật khẩu user

# --- Kết nối Odoo qua XML-RPC ---
st.title("Odoo ML App")

try:
    common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common")
    uid = common.authenticate(db, username, password, {})
    if uid is None:
        st.error("Authentication failed. Kiểm tra username/password/database")
        st.stop()
    
    models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object")
except Exception as e:
    st.error(f"Không thể kết nối tới Odoo: {e}")
    st.stop()

# --- Lấy dữ liệu Sales Orders ---
try:
    sales_orders = models.execute_kw(
        db, uid, password,
        'sale.order', 'search_read',
        [[]], {'fields': ['name', 'amount_total']}
    )
except Exception as e:
    st.error(f"Lỗi khi lấy dữ liệu từ Odoo: {e}")
    st.stop()

if not sales_orders:
    st.warning("Không có Sales Orders nào để hiển thị")
    st.stop()

df = pd.DataFrame(sales_orders)
st.subheader("Sales Orders từ Odoo")
st.dataframe(df)

# --- Load mô hình ML ---
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Không tìm thấy file model.pkl. Hãy chắc chắn model tồn tại.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi load model: {e}")
    st.stop()

# --- Dự đoán giá trị ---
# Lưu ý: cột phải giống với khi bạn train model
feature_columns = ['amount_total']  # thay bằng cột đúng nếu model khác
try:
    df['predicted_amount'] = model.predict(df[feature_columns])
except Exception as e:
    st.error(f"Lỗi khi dự đoán: {e}")
    st.stop()

st.subheader("Sales Orders với dự đoán từ ML Model")
st.dataframe(df)
