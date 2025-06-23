import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    det_curve,
)
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import textstat
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from typing import List, Dict, Any, Tuple

# --- 1. INITIAL SETUP AND CALCULATION FUNCTIONS ---

def initialize_benchmark_models():
    """Downloads and loads the necessary models for benchmarks."""
    print("Initializing models for benchmarks (this may take a while)...")
    try:
        stopwords_es = nltk.corpus.stopwords.words('spanish')
    except LookupError:
        nltk.download('stopwords')
        stopwords_es = nltk.corpus.stopwords.words('spanish')

    # GPT-2 model for perplexity
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("Models ready.")
    return tokenizer, model, stopwords_es

def compute_perplexity(text: str, tokenizer, model) -> float:
    """Computes the perplexity of a given text."""
    if not text or not text.strip():
        return float('nan')
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        return np.exp(loss)
    except Exception as e:
        print(f"Error computing perplexity: {e}")
        return float('nan')

def compute_semantic_coherence(texts: List[str], stopwords_es: List[str]) -> float:
    """Computes the average semantic coherence of a list of texts."""
    if len(texts) < 2:
        return 1.0  # If 0 or 1 text, coherence is maximal by definition.
    
    vectorizer = TfidfVectorizer(stop_words=stopwords_es)
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        # Compute similarity only between unique pairs of documents
        similarities = cosine_similarity(tfidf_matrix)
        # Get indices of the upper triangle of the matrix (excluding the diagonal)
        indices = np.triu_indices_from(similarities, k=1)
        if len(similarities[indices]) > 0:
            return np.mean(similarities[indices])
        return 0.0
    except ValueError:
        # This can happen if all texts are empty or only contain stopwords
        return 0.0




# --- 2. MAIN BENCHMARK ANALYSIS FUNCTION ---

def analyze_benchmarks_from_json(
    json_path: str,
    models_to_evaluate: List[str],
) -> List[Dict[str, Any]]:
    """
    Extracts text from a JSON file, computes text quality benchmarks per model,
    and returns the results.

    Args:
        json_path (str): Path to the consolidated JSON file.
        models_to_evaluate (List[str]): List of model suffixes (e.g., ["DS", "gpt4"]).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with benchmark results per model.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at '{json_path}'")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file at '{json_path}' is not valid JSON.")
        return []

    tokenizer, model, stopwords_es = initialize_benchmark_models()
    final_results = []

    print("\nStarting benchmark analysis by model...")
    for model_name in models_to_evaluate:
        print(f"Processing model: {model_name}...")
        label_key = f"label_{model_name}"
        model_texts = []

        # Collect all "respuesta_llm" texts for the current model
        for region, sections in json_data.items():
            for section in sections:
                if label_key in section and "respuesta_llm" in section[label_key]:
                    text = section[label_key]["respuesta_llm"]
                    if text and text.strip():
                        model_texts.append(text)

        if not model_texts:
            print(f"No texts found for model '{model_name}'. Skipping.")
            continue

        # Compute benchmarks for the model's text collection
        readability_scores = [textstat.flesch_reading_ease(t) for t in model_texts]
        perplexity_scores = [compute_perplexity(t, tokenizer, model) for t in model_texts]

        # Filter out NaNs before computing the mean
        readability_scores = [s for s in readability_scores if not np.isnan(s)]
        perplexity_scores = [s for s in perplexity_scores if not np.isnan(s)]

        mean_readability = np.mean(readability_scores) if readability_scores else 0
        mean_perplexity = np.mean(perplexity_scores) if perplexity_scores else 0
        mean_coherence = compute_semantic_coherence(model_texts, stopwords_es)

        final_results.append({
            "Model": model_name,
            "Readability (Flesch)": mean_readability,
            "Perplexity": mean_perplexity,
            "Semantic Coherence": mean_coherence
        })

    return show_benchmark_results(final_results)


# --- 3. FUNCTION TO DISPLAY RESULTS ---

def show_benchmark_results(results: List[Dict[str, Any]]):
    """Displays benchmark results in a table and a bar chart."""
    if not results:
        print("No results to display.")
        return

    # Use pandas for a clearer table
    df = pd.DataFrame(results).set_index("Model")
    print("\n--- Text Quality Benchmark Results by Model ---")
    print(df.round(2))

    # Bar chart
    df.plot(kind='bar', figsize=(12, 7), rot=0)
    plt.title("Text Quality Benchmark Comparison by Model")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.legend(title="Metrics")
    plt.show()





import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import List, Dict, Any

def analyze_stability(
    folder_path: str,
    models_to_evaluate: List[str]
) -> List[Dict[str, Any]]:
    """
    Analyzes the stability and self-consistency of models based on multiple
    result JSON files located in a folder.

    Args:
        folder_path (str): Path to the folder containing the JSON files.
        models_to_evaluate (List[str]): List of model names to analyze (e.g., ["DS", "gpt4"]).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with results per model.
    """
    # --- Step 1: Aggregate data from all JSON files ---
    # Structure: {clause_text: {model_name: [resp1, resp2, ...]}}
    grouped_data = defaultdict(lambda: defaultdict(list))

    try:
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        if not json_files:
            print(f"Error: No .json files found in folder '{folder_path}'")
            return []
        print(f"Found {len(json_files)} JSON files to analyze.")
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []

    for file_name in json_files:
        full_path = os.path.join(folder_path, file_name)
        with open(full_path, 'r', encoding='utf-8') as f:
            run_data = json.load(f)
            for region, sections in run_data.items():
                for section in sections:
                    key_text = section.get("text", "").strip()
                    if not key_text:
                        continue

                    for model_name in models_to_evaluate:
                        label_key = f"label_{model_name}"
                        if label_key in section and "respuesta_llm" in section[label_key]:
                            response = section[label_key]["respuesta_llm"]
                            if response and response.strip():
                                grouped_data[key_text][model_name].append(response)

    # --- Step 2: Compute metrics per model ---
    final_results = []
    print("\nCalculating stability and consistency per model...")
    for model_name in models_to_evaluate:
        stability_per_clause = []
        consistency_per_clause = []

        # Iterate over each group of responses for a unique clause
        for grouped_responses in grouped_data.values():
            model_responses = grouped_responses.get(model_name, [])

            # Need at least 2 responses to compare
            if len(model_responses) < 2:
                continue

            try:
                vectorizer = TfidfVectorizer().fit_transform(model_responses)
                similarity_matrix = cosine_similarity(vectorizer)
                distances = 1 - similarity_matrix[np.triu_indices(len(model_responses), k=1)]

                if len(distances) == 0:
                    continue

                # Stability: 1 minus the normalized average distance
                max_dist = np.max(distances)
                stability = 1 - (np.mean(distances) / max_dist if max_dist > 0 else 0)

                # Consistency: 1 minus the proportion of contradictions
                contradictions = np.sum(distances > 0.5)
                consistency = 1 - (contradictions / len(distances))

                stability_per_clause.append(stability)
                consistency_per_clause.append(consistency)
            except ValueError:
                continue

        # Average metrics over all clauses for this model
        if stability_per_clause:
            mean_stability = np.mean(stability_per_clause)
            mean_consistency = np.mean(consistency_per_clause)

            final_results.append({
                "Model": model_name,
                "Stability": mean_stability,
                "Self-Consistency": mean_consistency
            })
        else:
            print(f"Could not compute metrics for model '{model_name}'. "
                  "Make sure you have at least 2 runs with responses for this model.")

    return show_stability_results(final_results)


def show_stability_results(results: List[Dict[str, Any]]):
    """Displays stability and self-consistency results in a table and a chart."""
    if not results:
        print("No stability results to display.")
        return

    df = pd.DataFrame(results).set_index("Model")
    print("\n--- Stability and Self-Consistency Results ---")
    print(df.round(4))

    # Bar chart
    df.plot(kind='bar', figsize=(12, 7), rot=0)
    plt.title("Stability and Self-Consistency per Model (average across multiple runs)")
    plt.ylabel("Score (0 to 1)")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.show()



def compute_metrics_from_json(
    json_path: str,
    models_to_evaluate: List[str],
    categories: List[str]
) -> Tuple[Dict, Dict, Dict]:
    """
    Computes classification metrics from a JSON file containing expected labels
    and predictions from multiple models.

    Args:
        json_path (str): Path to the input JSON file.
        models_to_evaluate (List[str]): List of model names to evaluate
                                        (e.g., ["gpt4", "DS", "GPT4o"]).
        categories (List[str]): List of label categories to evaluate
                                (e.g., ["legalidad", "abusividad"]).

    Returns:
        Tuple[Dict, Dict, Dict]: A tuple containing:
            - A dictionary with metrics (Precision, Recall, etc.) per model.
            - A dictionary with ROC curve data per model.
            - A dictionary with DET curve data per model.
    """

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at path: '{json_path}'")
        return {}, {}, {}  # Return empty tuples on error
    except json.JSONDecodeError:
        print(f"Error: The file '{json_path}' is not valid JSON or is empty.")
        return {}, {}, {}

    metrics_by_model = {}
    roc_curves = {}
    det_curves = {}

    for model in models_to_evaluate:
        y_true = []
        y_pred = []
        label_key_pred = f"label_{model}"

        for contract, sections in json_data.items():
            for section in sections:
                label_expected = section.get("label_expected")
                label_predicted = section.get(label_key_pred)

                # Skip this section if either expected or predicted label is missing
                if not label_expected or not label_predicted:
                    continue

                # Skip if all expected labels are empty (non-informative section)
                if all(label_expected.get(cat, "") == "" for cat in categories):
                    continue

                if all(label_predicted.get(cat, "") == "" for cat in categories):
                    continue



                for category in categories:
                    true_value = label_expected.get(category, "")
                    predicted_value = label_predicted.get(category, "")

                    # Convert labels to binary format (1 for positive, 0 for negative)
                    if category == "legalidad":
                        y_true.append(1 if true_value == "L" else 0)
                        y_pred.append(1 if predicted_value == "L" else 0)
                    else:  # For abusividad, riesgo, vaguedades, lagunas
                        y_true.append(1 if true_value == "S" else 0)
                        y_pred.append(1 if predicted_value == "S" else 0)

        # If we have data for this model, compute metrics
        if y_true and y_pred:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # AUC and curves can only be computed if both classes are present
            try:
                auc = roc_auc_score(y_true, y_pred)
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                fpr_det, fnr_det, _ = det_curve(y_true, y_pred)
                roc_curves[model] = (fpr, tpr)
                det_curves[model] = (fpr_det, fnr_det)
            except ValueError:
                print(f"Warning: Cannot compute AUC/ROC/DET for model '{model}'. "
                      "Only one class was found in the data.")
                auc = float('nan')
                roc_curves[model] = (None, None)
                det_curves[model] = (None, None)

            metrics_by_model[model] = {
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "AUC": auc,
            }

    return show_results(metrics_by_model, roc_curves, det_curves)


def show_results(metrics: Dict, roc_curves: Dict, det_curves: Dict):
    """
    Displays a metrics table and plots ROC and DET curves.

    Args:
        metrics (Dict): Dictionary with metrics per model.
        roc_curves (Dict): Dictionary with ROC curve data per model.
        det_curves (Dict): Dictionary with DET curve data per model.
    """
    if not metrics:
        print("No metric data found to display.")
        return

    # --- Display metrics table ---
    df = pd.DataFrame.from_dict(metrics, orient="index")
    print("\n📊 METRICS BY MODEL:\n")
    print(df.round(3))

    # --- Plot ROC Curve ---
    plt.figure(figsize=(8, 6))
    for model, (fpr, tpr) in roc_curves.items():
        if fpr is not None and tpr is not None:
            plt.plot(fpr, tpr, label=f"{model} (AUC = {metrics[model]['AUC']:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curves by Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot DET Curve ---
    plt.figure(figsize=(8, 6))
    for model, (fpr_det, fnr_det) in det_curves.items():
        if fpr_det is not None and fnr_det is not None:
            plt.plot(fpr_det, fnr_det, label=f"{model}")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.title("DET Curves by Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
