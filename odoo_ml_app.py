# odoo_ml_app.py
import xmlrpc.client
import pandas as pd
import pickle
import streamlit as st
from sklearn.linear_model import LinearRegression

# --------------------------
# Cấu hình Odoo
# --------------------------
url = "http://localhost:8069"
db = "Odoo"
username = "23070615@vnu.edu.vn"         # Administrator user
password = "Dothiphuong99" # Thay bằng password admin thật

# --------------------------
# Kết nối Odoo qua XML-RPC
# --------------------------
common = xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/common')
uid = common.authenticate(db, username, password, {})
if not uid:
    st.error("Đăng nhập thất bại. Kiểm tra username/password.")
    st.stop()

models = xmlrpc.client.ServerProxy(f'{url}/xmlrpc/2/object')

# --------------------------
# Lấy dữ liệu Sale Order
# --------------------------
sale_orders = models.execute_kw(
    db, uid, password,
    'sale.order', 'search_read',
    [[]],
    {'fields': ['name','partner_id','amount_total','date_order'], 'limit': 50}
)

df_sales = pd.DataFrame(sale_orders)
if not df_sales.empty and 'partner_id' in df_sales.columns:
    df_sales['partner_name'] = df_sales['partner_id'].apply(lambda x: x[1] if isinstance(x, list) else x)
    df_sales.drop(columns=['partner_id'], inplace=True)

# --------------------------
# Lấy dữ liệu Inventory / Stock Picking
# --------------------------
stock_pickings = models.execute_kw(
    db, uid, password,
    'stock.picking', 'search_read',
    [[]],
    {'fields': ['name','partner_id','state','scheduled_date'], 'limit': 50}
)

df_stock = pd.DataFrame(stock_pickings)
if not df_stock.empty and 'partner_id' in df_stock.columns:
    df_stock['partner_name'] = df_stock['partner_id'].apply(lambda x: x[1] if isinstance(x, list) else x)
    df_stock.drop(columns=['partner_id'], inplace=True)

# --------------------------
# Tạo model ML demo (Linear Regression)
# --------------------------
# Nếu bạn chưa có model.pkl, tạo model test
try:
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    X = df_sales[['amount_total']] if 'amount_total' in df_sales.columns else pd.DataFrame([[0],[1],[2]])
    y = X * 1.1  # giả lập prediction
    model = LinearRegression()
    model.fit(X,y)
    with open("model.pkl","wb") as f:
        pickle.dump(model,f)

# --------------------------
# Chạy dự đoán
# --------------------------
if 'amount_total' in df_sales.columns:
    X = df_sales[['amount_total']]
    df_sales['prediction'] = model.predict(X)

# --------------------------
# Hiển thị bằng Streamlit
# --------------------------
st.title("Odoo ML Integration App")

st.subheader("Sales Orders")
st.dataframe(df_sales)

st.subheader("Inventory / Stock Picking")
st.dataframe(df_stock)
