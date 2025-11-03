import os
import shutil
import tempfile
import time
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "OpenSource_GenAI_Summarization")
HF_TOKEN = os.getenv("HF_TOKEN", None)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Get experiment ID
exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise SystemExit(f"Experiment named '{EXPERIMENT_NAME}' not found. Run experiments first.")
EXP_ID = exp.experiment_id

# Choose metric by which to select best run
METRIC_TO_OPTIMIZE = "overall"  # changeable
ORDER = "desc"  # 'desc' for larger-is-better, 'asc' otherwise

def find_best_run(experiment_id: str, metric_name: str, order: str = "desc"):
    runs = client.search_runs([experiment_id], filter_string="", run_view_type=1, max_results=1000)
    # Filter runs that have the metric
    filtered = [r for r in runs if metric_name in r.data.metrics]
    if not filtered:
        raise SystemExit(f"No runs with metric '{metric_name}' found under experiment id {experiment_id}.")
    # Sort
    reverse = True if order == "desc" else False
    best = sorted(filtered, key=lambda r: r.data.metrics[metric_name], reverse=reverse)[0]
    return best

# 1) Find best run
best_run = find_best_run(EXP_ID, METRIC_TO_OPTIMIZE, ORDER)
best_run_id = best_run.info.run_id
print(f"Selected best run {best_run_id} using metric '{METRIC_TO_OPTIMIZE}' = {best_run.data.metrics[METRIC_TO_OPTIMIZE]}")

# 2) Inspect artifact path for the generated model files
MODEL_ARTIFACT_NAME = "summarizer_model_pyfunc"  # artifact name to log

# Decide base model name from run params (if logged)
base_model = best_run.data.params.get("model_name", "facebook/bart-large-cnn")
print(f"Base model inferred: {base_model}")

# Create a local directory to persist tokenizer+model
local_model_dir = Path(tempfile.mkdtemp(prefix="mlflow_shot_"))
print(f"Will save model files to: {local_model_dir}")

# download/save model and tokenizer to local_model_dir
print("Downloading/saving model and tokenizer to local path (this may take time)...")
token = HF_TOKEN or None
AutoTokenizer.from_pretrained(base_model, cache_dir=None, use_auth_token=token).save_pretrained(local_model_dir)
AutoModelForSeq2SeqLM.from_pretrained(base_model, cache_dir=None, use_auth_token=token).save_pretrained(local_model_dir)
print("Saved pretrained model to local directory.")

# 3) Define pyfunc wrapper class
import mlflow.pyfunc
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SummarizerPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # model_dir is provided as an artifact (local path)
        model_dir = context.artifacts["model_dir"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.to(self.device)
    def predict(self, context, model_input):
        # model_input: DataFrame with a 'text' column (or 'prompt' etc)
        texts = model_input["text"].tolist()
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model.generate(**inputs, max_length=128)
        dec = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        import pandas as pd
        return pd.DataFrame({"summary": dec})

# 4) Log pyfunc model under the best run (create a temporary run or reuse existing run)
print("Logging pyfunc model as an artifact under the best run...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.start_run(run_id=best_run_id)  # attach to the best run
try:
    mlflow.pyfunc.log_model(
        artifact_path=MODEL_ARTIFACT_NAME,
        python_model=SummarizerPyfunc(),
        artifacts={"model_dir": str(local_model_dir)}
    )
    model_uri = f"runs:/{best_run_id}/{MODEL_ARTIFACT_NAME}"
    print(f"Pyfunc model logged to {model_uri}")
finally:
    mlflow.end_run()

# 5) Register model in registry
MODEL_REGISTRY_NAME = "summarizer-model"
print(f"Registering model '{MODEL_REGISTRY_NAME}' from {model_uri} ...")
try:
    registered_model = client.get_registered_model(MODEL_REGISTRY_NAME)
    print("Registered model already exists.")
except Exception:
    client.create_registered_model(MODEL_REGISTRY_NAME)
    print("Created new registered model.")

mv = client.create_model_version(name=MODEL_REGISTRY_NAME, source=model_uri, run_id=best_run_id)
print(f"Created model version {mv.version} for '{MODEL_REGISTRY_NAME}'")

# 6) Transition to Staging (optional)
print("Promoting to 'Staging' ...")
client.transition_model_version_stage(name=MODEL_REGISTRY_NAME, version=mv.version, stage="Staging", archive_existing_versions=False)
print(f"Model {MODEL_REGISTRY_NAME} version {mv.version} promoted to Staging")

# Cleanup temp saved model_dir if you want to remove local copy:
shutil.rmtree(local_model_dir)

print("Done. Visit MLflow UI and the Model Registry to inspect the registered model.")
