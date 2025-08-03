import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ========== Settings ==========
sns.set(style="whitegrid")
os.makedirs('report/figures', exist_ok=True)


# ========== Load and Clean ==========
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['Station Names', 'YEAR', 'Month', 'Max Temp', 'Min Temp',
             'Relative Humidity', 'Rainfall']].copy()
    df.columns = ['station', 'year', 'month', 'max_temp', 'min_temp', 'humidity', 'rainfall']
    df.dropna(inplace=True)
    df = df[(df['year'] >= 2010) & (df['year'] <= 2012)]
    return df


df = load_and_clean_data('data/65 years Weather Data Bangladesh.csv')

# ========== Features and Target ==========
X = df.drop(['rainfall', 'year'], axis=1)
y = df['rainfall']

# ========== Preprocessing ==========
categorical = ['station']
numerical = ['month', 'max_temp', 'min_temp', 'humidity']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical),
    ('num', StandardScaler(), numerical)
])

# ========== Pipelines ==========
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Train and Evaluate ==========
for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'Pred': y_pred
    }

# Random Forest Cross-Validation
rf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_scores = cross_val_score(rf_pipeline, X, y, cv=5, scoring='r2')
results['Random Forest']['CrossVal R2'] = rf_scores.mean()

# ========== Results Summary ==========
print("\n📊 Model Evaluation Summary:")
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        if metric != 'Pred':
            print(f"{metric}: {value:.2f}")

# ========== Plot Actual vs Predicted ==========
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Actual', color='black', linewidth=2)
for name, result in results.items():
    plt.plot(result['Pred'][:100], label=f'{name} Prediction')
plt.title('📈 Actual vs Predicted Rainfall')
plt.xlabel('Sample Index')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.tight_layout()
plt.savefig('report/figures/Actual_vs_Predicted_AllModels.png')
plt.show()
