# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load the dataset
df = pd.read_csv('laptopPrice.csv')

# Data Exploration
print(df.info())  # Overview of dataset
print(df.describe())  # Summary statistics

# Checking for missing values
print(df.isnull().sum())

# Visualizing some key relationships
plt.figure(figsize=(10,6))
sns.barplot(x='brand', y='Price', data=df)
plt.title('Laptop Brand vs Price')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='ram_gb', y='Price', hue='processor_brand', data=df)
plt.title('RAM Size vs Price by Processor Brand')
plt.show()

# Feature Engineering
# Handling categorical variables with OneHotEncoding
categorical_cols = ['brand', 'processor_brand', 'processor_name', 'ram_type', 'os', 'Touchscreen', 'msoffice']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_categorical_data = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_cols))

# Merging encoded data with original dataframe and dropping original categorical columns
df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

# Function to remove non-numeric characters (like 'GB', 'stars', 'bit')
def clean_numeric_column(column, unit=None):
    if unit:
        return df[column].str.replace(unit, '', regex=False).astype(float)
    else:
        return pd.to_numeric(df[column], errors='coerce')

# Clean columns with 'GB' and 'bit' units
df['ram_gb'] = clean_numeric_column('ram_gb', ' GB')
df['ssd'] = clean_numeric_column('ssd', ' GB')
df['hdd'] = clean_numeric_column('hdd', ' GB')
df['graphic_card_gb'] = clean_numeric_column('graphic_card_gb', ' GB')
df['os_bit'] = clean_numeric_column('os_bit', '-bit')

# Clean the rating column ('stars' suffix)
df['rating'] = clean_numeric_column('rating', ' star')
df['rating'] = clean_numeric_column('rating', 's')
df['rating'] = clean_numeric_column('rating', ' stars')

# Clean processor generation (handle 'Not Available')
df['processor_gnrtn'] = pd.to_numeric(df['processor_gnrtn'], errors='coerce')

# Now, check if the conversion was successful
print(df.head())


# Splitting the data into features (X) and target (y)
X = df.drop(['Price'], axis=1)
y = df['Price']

# Standardizing the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Selection and Training

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Model Evaluation

# Random Forest Evaluation
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

# Gradient Boosting Evaluation
gb_preds = gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_preds)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))

print(f'Random Forest MAE: {rf_mae}, RMSE: {rf_rmse}')
print(f'Gradient Boosting MAE: {gb_mae}, RMSE: {gb_rmse}')

# Cross-validation for model robustness
rf_cv_score = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
gb_cv_score = cross_val_score(gb_model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')

print(f'Random Forest Cross-validation Score: {-rf_cv_score.mean()}')
print(f'Gradient Boosting Cross-validation Score: {-gb_cv_score.mean()}')

# Visualization

# Plotting Actual vs Predicted Prices for Random Forest
plt.figure(figsize=(10,6))
plt.scatter(y_test, rf_preds, color='blue', alpha=0.6, label='Predicted vs Actual (Random Forest)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Fit Line')
plt.title('Random Forest: Predicted vs Actual Laptop Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()

# Plotting Actual vs Predicted Prices for Gradient Boosting
plt.figure(figsize=(10,6))
plt.scatter(y_test, gb_preds, color='green', alpha=0.6, label='Predicted vs Actual (Gradient Boosting)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Fit Line')
plt.title('Gradient Boosting: Predicted vs Actual Laptop Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()