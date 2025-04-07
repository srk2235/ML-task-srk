import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("c:\\Users\\rksri\\Downloads\\house_data_sample (2).csv")

# Encode categorical columns
le_mainroad = LabelEncoder()
df['Mainroad'] = le_mainroad.fit_transform(df['Mainroad'])  # yes=1, no=0

le_location = LabelEncoder()
df['Location'] = le_location.fit_transform(df['Location'])

# Features and target
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Mainroad', 'Parking', 'Location']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open("predict_model.pkl", "wb") as f:
    pickle.dump({
        'model': model,
        'le_mainroad': le_mainroad,
        'le_location': le_location
    }, f)

print("Model saved at:", os.path.abspath("predict_model.pkl"))
