import streamlit as st
import matplotlib.pyplot as plt
from predict import predict_training_time, suggest_optimization
from config import MODEL_COMPLEXITY, GPU_DATA

st.set_page_config(page_title="AI Carbon-Aware Trainer", layout="centered")

st.title("🌍 AI Carbon-Aware AI Trainer")
st.write("Predict energy and carbon footprint before training your model.")

# ---- USER INPUT ----
model_name = st.selectbox("Select Model", list(MODEL_COMPLEXITY.keys()))
dataset_size = st.slider("Dataset Size", 10000, 100000, 50000, step=5000)
epochs = st.slider("Epochs", 5, 50, 20)
batch_size = st.selectbox("Batch Size", [16, 32, 64, 128])
gpu_name = st.selectbox("Select GPU", list(GPU_DATA.keys()))

# ---- BUTTON ACTION ----
if st.button("Predict Carbon Impact 🚀"):

    # Predict current impact
    training_time, energy, co2 = predict_training_time(
        model_name,
        dataset_size,
        epochs,
        batch_size,
        gpu_name
    )

    # Results Section
    st.subheader("📊 Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("⏱ Training Time (hrs)", round(training_time, 2))
    col2.metric("⚡ Energy (kWh)", round(energy, 2))
    col3.metric("🌫 CO₂ (kg)", round(co2, 2))

    # Sustainability Score
    st.subheader("🌱 Sustainability Score")

    score = max(0, min(100, 100 - (co2 * 2)))
    st.metric("Green Score", f"{round(score, 1)}/100")
    st.progress(int(score))

    # Optimization Section
    st.subheader("🌱 Optimization Suggestion")

    current_co2, opt_co2, reduction, opt_batch, opt_epochs = suggest_optimization(
        model_name,
        dataset_size,
        epochs,
        batch_size,
        gpu_name
    )

    if reduction > 1:
        st.success(f"🚀 You can reduce emissions by {round(reduction, 2)}%")

        st.markdown(f"""
        ### Recommended Changes:
        - Increase batch size to **{opt_batch}**
        - Reduce epochs to **{opt_epochs}**
        - Optimized CO₂: **{round(opt_co2, 2)} kg**
        """)

        co2_saved = current_co2 - opt_co2
        st.info(f"If used across 1000 training runs, this saves approximately {round(co2_saved*1000,2)} kg of CO₂.")
        st.metric("🌍 CO₂ Saved per Training Run", f"{round(co2_saved, 2)} kg")

    else:
        st.info("Current configuration is already efficient.")

    # CO2 Comparison Chart
    st.subheader("📊 CO₂ Comparison")

    labels = ["Current CO₂", "Optimized CO₂"]
    values = [current_co2, opt_co2]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("CO₂ (kg)")
    ax.set_title("Before vs After Optimization")

    st.pyplot(fig)