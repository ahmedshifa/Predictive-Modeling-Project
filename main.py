import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Load dataset
data = pd.read_csv('predictive_maintenance_dataset.csv')

# Handle missing data (if any)
data = data.fillna(method='ffill')

# Convert date columns to datetime format and extract useful features if needed
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data = data.drop('date', axis=1)

# Check if there are other non-numeric columns and convert them
for col in data.columns:
    if data[col].dtype == 'object':
        try:
            data[col] = data[col].astype(float)
        except ValueError:
            data = pd.get_dummies(data, columns=[col])

# Separate features and labels
features = data.drop('failure', axis=1)
labels = data['failure']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple neural network model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)
