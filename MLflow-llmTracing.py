import os
import json
import mlflow
from dotenv import load_dotenv
from transformers import pipeline

# --- 1. Setup & Configuration ---
load_dotenv()

# Load MLflow tracking URI from .env or fallback to local
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("OpenSource_GenAI_Summarization")

print(f"✅ MLflow Tracking URI: {mlflow_tracking_uri}")
print("✅ MLflow Experiment: OpenSource_GenAI_Summarization")

hf_token = os.getenv("HF_TOKEN")

# MLflow experiment setup
if mlflow.active_run():
    mlflow.end_run()
mlflow.set_experiment("OpenSource_GenAI_Summarization")
print("✅ MLflow Experiment: OpenSource_GenAI_Summarization")

# Create a summarization pipeline using an open-source model
# (you can change model to "facebook/bart-large-cnn" or "t5-base" for faster runs)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", token=hf_token)

# --- 2. Define Prompts to Test ---
prompts_to_test = [
    {
        "name": "Standard Summary",
        "instruction": "Summarize the following article concisely."
    },
    {
        "name": "Bullet Point Summary",
        "instruction": "Summarize the article in 3-5 bullet points, focusing on main ideas."
    },
    {
        "name": "One Sentence Summary",
        "instruction": "Condense the article into one comprehensive sentence."
    },
    {
        "name": "Detailed Summary",
        "instruction": "Write a detailed summary capturing all major points and context."
    }
]


# --- 3. Mock Evaluation Function ---
def evaluate_summary(original_text: str, summary_text: str) -> dict:
    """Simple heuristic-based evaluation for demonstration."""
    eval_scores = {
        "conciseness": 0.0,
        "coherence": 0.0,
        "relevance": 0.0,
        "overall": 0.0,
    }

    orig_len = len(original_text.split())
    summ_len = len(summary_text.split())

    # Conciseness score (shorter summaries rated higher, within range)
    if summ_len < 0.2 * orig_len:
        eval_scores["conciseness"] = 9
    elif summ_len < 0.4 * orig_len:
        eval_scores["conciseness"] = 7
    else:
        eval_scores["conciseness"] = 5

    # Coherence: rough proxy using sentence structure
    sentence_count = summary_text.count(".")
    eval_scores["coherence"] = 8 if 1 < sentence_count < 5 else 6

    # Relevance: keywords overlap
    keywords = ["AI", "climate", "work", "India", "innovation"]
    overlap = sum(1 for k in keywords if k.lower() in summary_text.lower())
    eval_scores["relevance"] = (overlap / len(keywords)) * 10

    eval_scores["overall"] = sum(eval_scores.values()) / 3
    return eval_scores


# --- 4. Load Sample Articles ---
articles_dir = "sample_articles"
if not os.path.exists(articles_dir):
    raise FileNotFoundError(f"❌ Directory '{articles_dir}' not found. Please add your .txt articles there.")

article_files = [f for f in os.listdir(articles_dir) if f.endswith(".txt")]
if not article_files:
    raise ValueError(f"⚠️ No .txt files found in {articles_dir}/")

print(f"📄 Found {len(article_files)} articles to summarize.\n")


# --- 5. Run MLflow Experiments ---
for article_file in article_files:
    with open(os.path.join(articles_dir, article_file), "r", encoding="utf-8") as f:
        article_text = f.read()

    print(f"\n=== Processing {article_file} ===")

    for prompt in prompts_to_test:
        prompt_name = prompt["name"]
        instruction = prompt["instruction"]

        print(f"\n--- Running Prompt: {prompt_name} ---")

        with mlflow.start_run(run_name=f"{article_file}_{prompt_name}"):
            mlflow.log_param("article_file", article_file)
            mlflow.log_param("prompt_type", prompt_name)
            mlflow.log_param("instruction", instruction)
            mlflow.log_param("model_name", "facebook/bart-large-cnn")

            # Generate summary
            try:
                full_input = f"{instruction}\n\n{article_text}"
                result = summarizer(full_input, max_length=250, min_length=50, do_sample=False)
                summary = result[0]["summary_text"]
                print(f"🧾 Summary Generated ({prompt_name}):\n{summary[:300]}...\n")

                # Save & log generated summary
                out_path = f"generated_{article_file.replace('.txt','')}_{prompt_name.replace(' ','_')}.txt"
                with open(out_path, "w", encoding="utf-8") as f_out:
                    f_out.write(summary)
                mlflow.log_artifact(out_path)

                # Evaluate & log metrics
                eval_scores = evaluate_summary(article_text, summary)
                for metric, val in eval_scores.items():
                    mlflow.log_metric(metric, val)

                mlflow.log_metric("summary_length", len(summary.split()))
                mlflow.set_tag("status", "success")

            except Exception as e:
                print(f"❌ Error generating summary: {e}")
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error_message", str(e))
                continue

print("\n✅ All summarization runs complete! View results in MLflow UI.")
