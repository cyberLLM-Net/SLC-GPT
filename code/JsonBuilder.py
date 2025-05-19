import os
import json
import re
import subprocess
from docx import Document

def extract_clauses(doc_path):
    doc = Document(doc_path)
    clauses = []
    current_clause = ""
    
    # Regular expression to detect clause titles
    clause_title_regex = re.compile(
        r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
        r"ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH|FIFTEENTH|SIXTEENTH|"
        r"SEVENTEENTH|EIGHTEENTH|NINETEENTH|TWENTIETH|TWENTY-FIRST|TWENTY-SECOND|"
        r"TWENTY-THIRD|TWENTY-FOURTH|TWENTY-FIFTH|TWENTY-SIXTH|TWENTY-SEVENTH|"
        r"TWENTY-EIGHTH|TWENTY-NINTH|THIRTIETH)\.?",
        re.IGNORECASE
    )

    def new_clause(text):
        return {
            "text": text.strip(),
            "label_expected": {
                "legality": "",
                "abusiveness": "",
                "risk_areas": "",
                "vagueness": "",
                "gaps": ""
            },
            "label_gpt4": {
                "legality": "",
                "abusiveness": "",
                "risk_areas": "",
                "vagueness": "",
                "gaps": "",
                "alternative_drafting": ""
            },
            "label_DS": {
                "legality": "",
                "abusiveness": "",
                "risk_areas": "",
                "vagueness": "",
                "gaps": "",
                "alternative_drafting": ""
            },
            "label_GPT4o": {
                "legality": "",
                "abusiveness": "",
                "risk_areas": "",
                "vagueness": "",
                "gaps": "",
                "alternative_drafting": ""
            },
            "label_GPT35": {
                "legality": "",
                "abusiveness": "",
                "risk_areas": "",
                "vagueness": "",
                "gaps": "",
                "alternative_drafting": ""
            }
        }

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        if clause_title_regex.match(text):
            if current_clause:
                clauses.append(new_clause(current_clause))
            current_clause = text
        else:
            current_clause += " " + text

    if current_clause:
        clauses.append(new_clause(current_clause))

    return clauses
    
def process_documents(directory):
    results = {}

    for file in os.listdir(directory):
        if file.endswith(".docx"):
            full_path = os.path.join(directory, file)
            clauses = extract_clauses(full_path)
            results[file] = clauses

    # Save results to a JSON file
    with open("contracts.json", "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)

    print("✅ Results saved in 'contracts.json'.")
    
# Path where the .docx files are located
docx_directory = "../dataset/contractExamples_org"

# Run the clause extraction
process_documents(docx_directory)

import json

def clean_incomplete_labels(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_labels = [
        "label_expected",
        "label_gpt4",
        "label_GPT4o",
        "label_GPT35",
        "label_DS"
    ]

    total_filtered = 0

    for doc, clauses in data.items():
        for i, clause in enumerate(clauses):
            has_all = all(
                label in clause and clause[label] is not None
                for label in required_labels
            )

            if not has_all:
                # Keep only the "text" field
                text = clause.get("text", "")
                data[doc][i] = {"text": text}
                total_filtered += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned clauses: {total_filtered}")
    print(f"📝 Saved to: {output_path}")

clean_incomplete_labels("summary.json", "filtered_contracts.json")