import json
import mlflow

# Registrar la URI de SQLite de tu entorno
# move_chsa-triage-sft-metrics-fake_jston_to_mlflow_db.py

mlflow.set_tracking_uri("sqlite:////home/rafael/.chsa-triage/mlflow.db")
mlflow.set_experiment("qwen-sft-experiment")

# Cargar las métricas del archivo JSON / JSONL
archivo_json = "chsa-triage-sft-metrics-faken"  # Ajusta la ruta a tu archivo

with mlflow.start_run(run_name="qwen2.5-sft-run"):
    with open(archivo_json, "r") as f:
        contenido = f.read().replace("}{", "}\n{")
        
        for linea in contenido.splitlines():
            if not linea.strip():
                continue
            d = json.loads(linea)
            
            # MLflow requiere la estampa de tiempo en milisegundos (int)
            timestamp_ms = int(d["horodatage"] * 1000) if "horodatage" in d else None
            
            mlflow.log_metric(
                key=d["nom"],
                value=d["valeur"],
                step=d["etape"],
                timestamp=timestamp_ms
            )

print("¡Métricas importadas con éxito a MLflow!")