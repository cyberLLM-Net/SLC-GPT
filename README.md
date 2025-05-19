# SLC-GPT
SLC-GPT Repository
# 📝 Contract Analysis Pipeline

This repository contains all the components needed to process rental contracts in `.docx` format, label them, analyze them with different language models, and evaluate the outputs with appropriate metrics.

---

## 📁 Project Structure

### `original_dataset/`
Contains the original contract documents in `.docx` format before processing.

### `code/`
Includes the core Python scripts used in the processing and analysis pipeline:

- **`JsonBuilder.py`**  
  Converts raw `.docx` contract files into structured, labeled JSON format.

- **`queryAPI.py`**  
  Sends clause-by-clause queries to various LLMs (Large Language Models) to generate:
  - Original model-generated reports → `informes/`
  - Prompt-adjusted reports → `informes_cambio_prompt/`
  - Reports with alternative wording proposals → `redacción_alternativa/`

- **`metrix.py`**  
  Computes evaluation metrics such as Precision, Recall, F1-score, and AUC. Also generates ROC and DET curves to compare model performance.

### `JSON/`
Stores the final processed and labeled JSON files after analysis and cleaning.

---

## ⚙️ Usage

1. **Preprocess DOCX Contracts**
   ```bash
   python JsonBuilder.py
