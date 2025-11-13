import streamlit as st
import pandas as pd
import pickle
import xmlrpc.client
from sklearn.linear_model import LinearRegression

st.title("Odoo ML App (Fixed fields)")

# --- Cấu hình Odoo ---
url = "https://brenton-chevronny-kristi.ngrok-free.dev"  # Odoo online hoặc ngrok
db = "Odoo"           
username = "23070615@vnu.edu.vn"       
password = "Dothiphuong99"    

# --- Kết nối Odoo ---
try:
    common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common")
    uid = common.authenticate(db, username, password, {})
    if uid is None:
        st.error("Authentication failed")
        st.stop()
    models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object")
except Exception as e:
    st.error(f"Cannot connect to Odoo: {e}")
    st.stop()

# --- Lấy dữ liệu Sales Orders ---
try:
    # Lấy name, amount_total, số lượng sản phẩm trong order (order_line)
    sales_orders = models.execute_kw(
        db, uid, password,
        'sale.order', 'search_read',
        [[]], {'fields': ['name', 'amount_total', 'order_line']}
    )
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if not sales_orders:
    st.warning("No Sales Orders found")
    st.stop()

# --- Chuyển order_line thành số lượng sản phẩm ---
for so in sales_orders:
    so['num_products'] = len(so['order_line'])  # số lượng product lines

df = pd.DataFrame(sales_orders)
st.subheader("Sales Orders từ Odoo")
st.dataframe(df[['name','amount_total','num_products']])

# --- Train model luôn từ dữ liệu Odoo ---
feature_cols = ['num_products']
target_col = 'amount_total'

model = LinearRegression()
model.fit(df[feature_cols], df[target_col])

# Tùy chọn lưu model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

df['predicted_amount'] = model.predict(df[feature_cols])
st.subheader("Sales Orders với dự đoán ML")
st.dataframe(df[['name','amount_total','num_products','predicted_amount']])


