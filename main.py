import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("laptopPrice.csv")
data = data.drop(['rating'], axis=1)

# Data cleaning and preprocessing

# Extract numerical values from columns with units
data['ram_gb'] = data['ram_gb'].str.replace(' GB', '', regex=True).astype(int)
data['ssd'] = data['ssd'].str.replace(' GB', '', regex=True).astype(int)
data['hdd'] = data['hdd'].str.replace(' GB', '', regex=True).astype(int)

data.fillna(0, inplace=True)  # Replace missing values with 0

le = LabelEncoder()
categorical_cols = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_type', 'os', 'os_bit', 'graphic_card_gb', 'weight', 'warranty', 'Touchscreen', 'msoffice']
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

X = data.drop('Price', axis=1)
y = data['Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training and evaluation
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print("Linear Regression:")
print("MSE:", lr_mse)
print("R2 Score:", lr_r2)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print("Random Forest Regression:")
print("MSE:", rf_mse)
print("R2 Score:", rf_r2)

# Visualization
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()