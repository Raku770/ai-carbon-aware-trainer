import joblib
from config import MODEL_COMPLEXITY, GPU_DATA, EMISSION_FACTOR

# Load trained model
model = joblib.load("model/training_model.pkl")


def predict_training_time(model_name, dataset_size, epochs, batch_size, gpu_name):
    
    model_complexity = MODEL_COMPLEXITY[model_name]
    gpu_score = GPU_DATA[gpu_name]["score"]
    gpu_power = GPU_DATA[gpu_name]["power"]

    # Prepare input for model
    input_data = [[
        model_complexity,
        dataset_size,
        epochs,
        batch_size,
        gpu_score
    ]]

    # Predict training time
    training_time = model.predict(input_data)[0]

    # Calculate energy (kWh)
    energy = (gpu_power / 1000) * training_time

    # Calculate CO2 emission
    co2 = energy * EMISSION_FACTOR

    return training_time, energy, co2


# Test run (temporary test)
if __name__ == "__main__":
    time, energy, co2 = predict_training_time(
        model_name="ResNet50",
        dataset_size=50000,
        epochs=20,
        batch_size=32,
        gpu_name="RTX 3060"
    )

    print("Predicted Training Time (hours):", round(time, 2))
    print("Estimated Energy (kWh):", round(energy, 2))
    print("Estimated CO2 (kg):", round(co2, 2))

def suggest_optimization(model_name, dataset_size, epochs, batch_size, gpu_name):

    # Current prediction
    current_time, current_energy, current_co2 = predict_training_time(
        model_name, dataset_size, epochs, batch_size, gpu_name
    )

    # Try improved batch size (if small)
    optimized_batch = batch_size
    if batch_size < 64:
        optimized_batch = 64

    # Try reducing epochs slightly
    optimized_epochs = epochs
    if epochs > 20:
        optimized_epochs = int(epochs * 0.8)

    # Predict optimized result
    opt_time, opt_energy, opt_co2 = predict_training_time(
        model_name, dataset_size, optimized_epochs, optimized_batch, gpu_name
    )

    reduction_percent = ((current_co2 - opt_co2) / current_co2) * 100

    return current_co2, opt_co2, reduction_percent, optimized_batch, optimized_epochs