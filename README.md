# SLC-GPT  
SLC-GPT Repository  

# 📝 Contract Analysis Pipeline

This repository contains all the necessary components to process rental contracts in `.docx` format, label them, analyze them with different language models, and evaluate the results using appropriate metrics.

---

## 📁 Project Structure

### `dataset/`  
Contains the original contract documents in `.docx` format prior to processing.

### `library/`  
Includes the Python code used in the processing and analysis pipeline:

- **`JsonBuilder.py`**  
  Converts `.docx` contract files into a structured and labeled JSON format.

- **`queryAPI.py`**  
  Sends clause-by-clause queries to various LLMs (Large Language Models) to generate:  
  - Original model-generated reports.   
  - Prompt-adjusted reports.  
  - Reports with alternative redrafting proposals.

- **`metrix.py`**  
  Computes evaluation metrics such as Precision, Recall, F1-score, and AUC. Also generates ROC and DET curves to compare model performance.

---

## ⚙️ Usage: View **`library/README.md`** 
