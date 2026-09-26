# Motor Temperature Forecasting with PyTorch — MLP vs LSTM

A small AI Engineering project comparing **MLP and LSTM architectures** for short-term motor temperature forecasting.

The goal is to predict the motor temperature approximately **30 seconds into the future** using recent telemetry.

The project focuses less on building the most complex neural network and more on building a correct ML pipeline: temporal targets, leakage prevention, feature engineering, time-based validation, and model comparison.

---

## Problem

A monitoring system receives motor telemetry every second:

- motor voltage
- motor current
- RPM
- ambient temperature
- motor temperature

A simple monitoring system can trigger an alert only after the motor is already too hot.

Instead, the objective here is:

> Given the recent operating state of the motor, estimate its temperature ~30 seconds ahead.

---

## Dataset

The project uses a **synthetic telemetry dataset** designed to simulate different motor operating conditions.

Main columns:

```text
timestamp
motor_voltage_v
motor_current_a
rpm
ambient_temp_c
motor_temp_c
```

The dataset contains multiple load regimes, thermal inertia, varying ambient temperature, and overload periods.

---

## Feature Engineering

Besides the raw sensor values, a few physically meaningful features were created.

### Electrical power

```python
power_w = motor_voltage_v * motor_current_a
```

### Temperature momentum

Measures whether the motor temperature is currently rising or falling.

```python
temp_momentum = motor_temp_c.diff(periods=5)
```

### Recent average power

Represents recent motor load instead of relying only on instantaneous power.

```python
power_rolling_avg = power_w.rolling(window=10).mean()
```

The final feature set includes:

```text
motor_voltage_v
motor_current_a
power_w
rpm
ambient_temp_c
motor_temp_c
temp_momentum
power_rolling_avg
```

---

## Target

The model predicts future motor temperature.

```python
df["target_temp"] = df["motor_temp_c"].shift(-30)
```

This converts the forecasting problem into:

```text
recent telemetry
        ↓
temperature approximately 30 seconds later
```

---

## Temporal Validation

Because this is time-series data, the dataset is split chronologically rather than randomly.

```text
past ------------------------------------> future

TRAIN                         VALIDATION / TEST
```

This avoids training the model using information from future periods.

Input normalization is also fitted only on the training data and then applied to later data.

---

## Models

Two PyTorch architectures were compared using the same telemetry features.

### LSTM

The LSTM receives a sequence of recent telemetry and learns temporal relationships between load and temperature.

A 30-second input window performed significantly better than a shorter 10-second window, which is consistent with the thermal inertia of the system.

### MLP

A simpler feed-forward neural network was also tested using the temporal input representation.

Despite being simpler, the MLP produced better validation results.

---

## Results

| Model | MAE | RMSE | Max Error |
|---|---:|---:|---:|
| LSTM (30s window) | 0.3216 °C | 0.4498 °C | 3.8655 °C |
| **MLP** | **0.2223 °C** | **0.2845 °C** | **1.3190 °C** |

The MLP performed better across all three evaluated metrics.

This was an important result of the experiment:

> A more sophisticated architecture is not automatically a better engineering solution.

For this dataset, feature engineering and temporal context allowed a simpler model to outperform the recurrent architecture.

---

## Key Lessons

- Time-series targets must be aligned carefully.
- Random train/test splits can create misleading results for temporal data.
- Scalers should be fitted only using training data.
- Low training loss alone does not prove that a model generalizes.
- Physical knowledge can guide useful feature engineering.
- Thermal systems have inertia, so historical context matters.
- Always compare complex models against simpler alternatives.
- Model complexity should be justified by measurable improvement.

---

## Stack

```text
Python
PyTorch
Pandas
NumPy
scikit-learn
Matplotlib
```

---

## Running the Project

Clone the repository:

```bash
git clone <repository-url>
cd motor-temperature-forecasting
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebook:

```bash
jupyter notebook
```

---

## Project Structure

```text
motor-temperature-forecasting/
│
├── README.md
├── PyTorch.ipynb
├── requirements.txt
│
└── data/
    └── northstar_motor_telemetry.csv
```

---

## Notes

This project uses **synthetic motor telemetry** and is intended as an AI Engineering experiment.

The reported metrics should not be interpreted as performance guarantees for real industrial equipment.

A production system would require validation using real sensor data, multiple motors and operating environments, monitoring for data drift, and safety-specific evaluation around critical temperature thresholds.

---

## Main Takeaway

The most useful part of this experiment was not getting a neural network loss to decrease.

It was going through the full engineering loop:

```text
problem definition
      ↓
target construction
      ↓
feature engineering
      ↓
temporal validation
      ↓
LSTM
      ↓
MLP
      ↓
model comparison
      ↓
engineering decision
```

In this case, the simpler model won.
