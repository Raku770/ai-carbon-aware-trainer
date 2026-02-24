import numpy as np
import pandas as pd
from config import MODEL_COMPLEXITY, GPU_DATA

# Number of samples
NUM_SAMPLES = 800

data = []

for _ in range(NUM_SAMPLES):

    model = np.random.choice(list(MODEL_COMPLEXITY.keys()))
    model_complexity = MODEL_COMPLEXITY[model]

    dataset_size = np.random.randint(10000, 100000)
    epochs = np.random.randint(5, 50)
    batch_size = np.random.choice([16, 32, 64, 128])

    gpu = np.random.choice(list(GPU_DATA.keys()))
    gpu_score = GPU_DATA[gpu]["score"]

    # Training time formula
    noise = np.random.uniform(0.5, 2.0)

    training_time = (
        (model_complexity * dataset_size * epochs)
        / (gpu_score * batch_size * 1000)
    ) + noise

    data.append([
        model_complexity,
        dataset_size,
        epochs,
        batch_size,
        gpu_score,
        training_time
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "ModelComplexity",
    "DatasetSize",
    "Epochs",
    "BatchSize",
    "GPUScore",
    "TrainingTime"
])

# Save dataset
df.to_csv("data/synthetic_training_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Features and target
X = df[["ModelComplexity", "DatasetSize", "Epochs", "BatchSize", "GPUScore"]]
y = df["TrainingTime"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Model trained successfully!")
print(f"Mean Absolute Error: {mae:.4f}")

# Save model
joblib.dump(model, "model/training_model.pkl")

print("Model saved successfully!")