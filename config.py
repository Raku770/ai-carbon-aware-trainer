# Model Complexity Mapping
MODEL_COMPLEXITY = {
    "MobileNet": 1,
    "ResNet50": 2,
    "BERT-base": 3
}

# GPU Database (Power in Watts + Performance Score)
GPU_DATA = {
    "T4": {"power": 70, "score": 1},
    "RTX 3060": {"power": 170, "score": 2},
    "RTX 3090": {"power": 350, "score": 3}
}

# Carbon Emission Factor (India average)
EMISSION_FACTOR = 0.82  # kg CO2 per kWh