import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Tạo dữ liệu giả lập
df_sales = pd.DataFrame({'amount_total': [100, 200, 150, 300, 250]})

# Tạo mô hình
X = df_sales[['amount_total']]
y = df_sales['amount_total']
model = LinearRegression()
model.fit(X, y)

# Lưu mô hình
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model.pkl đã được tạo xong")
