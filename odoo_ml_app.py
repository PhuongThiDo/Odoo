
---

## **4️⃣ File `odoo_ml_app.py`**

```python
import streamlit as st
import pandas as pd
import pickle
import xmlrpc.client
from sklearn.linear_model import LinearRegression

st.title("Odoo ML App")

# --- Cấu hình Odoo ---
# Thay URL bằng Odoo online hoặc URL ngrok nếu local
url = "https://brenton-chevronny-kristi.ngrok-free.dev"  
db = "Odoo"           
username = "23070615@vnu.edu.vn"       
password = "Dothiphuong99"    

# --- Kết nối Odoo ---
try:
    common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common")
    uid = common.authenticate(db, username, password, {})
    if uid is None:
        st.error("Authentication failed. Kiểm tra username/password/database")
        st.stop()
    models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object")
except Exception as e:
    st.error(f"Cannot connect to Odoo: {e}")
    st.stop()

# --- Lấy dữ liệu Sales Orders ---
try:
    sales_orders = models.execute_kw(
        db, uid, password,
        'sale.order', 'search_read',
        [[]], {'fields': ['name', 'amount_total', 'discount', 'num_products']} 
    )
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if not sales_orders:
    st.warning("No Sales Orders found")
    st.stop()

df = pd.DataFrame(sales_orders)
st.subheader("Sales Orders từ Odoo")
st.dataframe(df)

# --- Train/load model ---
feature_cols = ['num_products', 'discount']
target_col = 'amount_total'

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.warning("Model chưa có, train Linear Regression từ dữ liệu Odoo")
    model = LinearRegression()
    model.fit(df[feature_cols], df[target_col])
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Đã train và lưu model.pkl")

# --- Dự đoán ---
df['predicted_amount'] = model.predict(df[feature_cols])

st.subheader("Sales Orders với dự đoán ML")
st.dataframe(df)
