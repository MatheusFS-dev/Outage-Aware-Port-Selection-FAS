from pathlib import Path
_CHAPTER = Path(__file__).resolve().parents[2]

import os
import sys

print("\n\033[91mRun this script with sudo if using CPU to measure model stats!!\033[0m")
print(f"\033[94m{sys.executable} is running this script\033[0m")
print(
    "\033[93mWARNING: If using Anaconda Python environment, ensure you are using the correct Python interpreter.\033[0m\n"
)

# sudo /home/matheus/anaconda3/envs/tf-optuna/bin/python measure_model_stats.py

import csv
from tensorflow.keras.models import load_model
from araras.ml.model.stats import get_model_usage_stats, get_memory_and_time, get_flops

# 1. map model names to saved‐model paths (adjust paths as needed)
model_paths = {
    "cnn_3_observed_ports": str(_CHAPTER / "results/models/cnn/optuna_study_3_ports/models/model.keras"),
    "cnn_4_observed_ports": str(_CHAPTER / "results/models/cnn/optuna_study_4_ports/models/model.keras"),
    "cnn_5_observed_ports": str(_CHAPTER / "results/models/cnn/optuna_study_5_ports/models/model.keras"),
    "cnn_6_observed_ports": str(_CHAPTER / "results/models/cnn/optuna_study_6_ports/models/model.keras"),
    "cnn_7_observed_ports": str(_CHAPTER / "results/models/cnn/optuna_study_7_ports/models/model.keras"),
    "cnn_10_observed_ports": str(_CHAPTER / "results/models/cnn/optuna_study_10_ports/models/model.keras"),
    "cnn_15_observed_ports": str(_CHAPTER / "results/models/cnn/optuna_study_15_ports/models/model.keras"),
}

output_dir = str(_CHAPTER / "runs/cnn_model_stats")
os.makedirs(output_dir, exist_ok=True)

# 3. group models by prefix (cnn, dnn, lstm, tcnn)
grouped = {}
for name, path in model_paths.items():
    prefix = name.split("_", 1)[0].upper()  # yields 'CNN','DNN','LSTM','TCNN'
    grouped.setdefault(prefix, {})[name] = path

# 4. common CSV header
header = [
    "Observed Ports",
    "Total params",
    "Size",
    "CPU memory",
    "CPU latency",
    "GPU memory",
    "GPU latency",
    "Avg power (CPU)",
    "Avg energy (CPU)",
    "Avg power (GPU)",
    "Avg energy (GPU)",
    "FLOPs",
]

# 5. iterate groups and write one CSV per model type
for model_type, paths in grouped.items():
    csv_path = os.path.join(output_dir, f"summary_{model_type}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for name, path in paths.items():
            model = load_model(path)

            # memory & latency
            mem_cpu, inf_cpu = get_memory_and_time(
                model, batch_size=1, device=-1, warmup_runs=10, test_runs=50
            )
            mem_gpu, inf_gpu = get_memory_and_time(
                model, batch_size=1, device=0, warmup_runs=10, test_runs=50
            )

            # power & energy
            _, avg_power_cpu, avg_energy_cpu = get_model_usage_stats(
                model, device=-1, n_trials=10000, verbose=True
            )
            _, avg_power_gpu, avg_energy_gpu = get_model_usage_stats(
                model, device=0, n_trials=10000, verbose=True
            )

            total_params = model.count_params()
            size_mb = os.path.getsize(path) / (1024**2)
            observed_ports = name.split("_")[2]

            writer.writerow(
                [
                    observed_ports,
                    total_params,
                    f"{size_mb:.2f} MB",
                    mem_cpu,
                    inf_cpu,
                    mem_gpu,
                    inf_gpu,
                    avg_power_cpu,
                    avg_energy_cpu,
                    avg_power_gpu,
                    avg_energy_gpu,
                    get_flops(model, batch_size=1),
                ]
            )

            print(f"✓ Processed {name}")

    print(f"✓ CSV for {model_type} saved to {csv_path}")
