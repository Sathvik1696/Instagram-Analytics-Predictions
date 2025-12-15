import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Data
file_path = 'Instagram_Analytics.csv'  # Assumes running from project root
print(f"Loading data from {file_path}...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# 2. Preprocessing
print("Preprocessing data...")

# Convert upload_date to datetime
df['upload_date'] = pd.to_datetime(df['upload_date'])

# Feature Engineering
df['upload_hour'] = df['upload_date'].dt.hour
df['upload_day_of_week'] = df['upload_date'].dt.dayofweek # 0=Monday, 6=Sunday
df['upload_month'] = df['upload_date'].dt.month

# Select Features and Target
# Features: Content measurable at upload time or controllable
features_to_use = [
    'media_type', 
    'caption_length', 
    'hashtags_count', 
    'traffic_source', 
    'content_category',
    'upload_hour', 
    'upload_day_of_week',
    'upload_month'
]
target = 'likes'

X = df[features_to_use]
y = df[target]

# Define Categorical and Numerical features
categorical_features = ['media_type', 'traffic_source', 'content_category']
numerical_features = ['caption_length', 'hashtags_count', 'upload_hour', 'upload_day_of_week', 'upload_month']

# Create Transformers
numerical_transformer = SimpleImputer(strategy='mean') # Handle missing values just in case
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Model Training and Evaluation Pipeline
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining models...")

for name, model in models.items():
    print(f"Training {name}...")
    
    # Create Pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2 Score: {r2:.4f}")
    
    # Print sample predictions
    print(f"  Example Predictions (Actual vs Predicted):")
    comparison = pd.DataFrame({'Actual': y_test[:5].values, 'Predicted': y_pred[:5].round(1)})
    print(comparison.to_string(index=False))
    print("-" * 30)

# 4. Comparison Visualization
print("\nGenerating comparison plot...")
results_df = pd.DataFrame(results).T

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot R2 Score
results_df['R2'].plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'], ax=axes[0])
axes[0].set_title('R2 Score (Higher is Better)')
axes[0].set_ylabel('R2 Score')
axes[0].axhline(0, color='black', linewidth=0.8)

# Plot MAE
results_df['MAE'].plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'], ax=axes[1])
axes[1].set_title('Mean Absolute Error (Lower is Better)')
axes[1].set_ylabel('MAE (Likes)')

plt.tight_layout()
plt.savefig('model_comparison.png')
print("Plot saved as 'model_comparison.png'")

print("\nDone!")

