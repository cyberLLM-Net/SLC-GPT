import os
import json
import re
from docx import Document
from typing import List, Dict

def json_builder(directory: str, output_json: str = "contratos.json") -> Dict[str, List[Dict]]:
    """
    Processes all .docx documents in a directory, extracting legal clauses
    and structuring them into JSON format.

    Args:
        directory (str): Path to the directory containing .docx files.
        output_json (str): Name of the output JSON file.

    Returns:
        Dict[str, List[Dict]]: Dictionary with filenames as keys
                               and lists of extracted clauses as values.
    """
    
    # Regular expression to detect clause titles
    regex_clause = re.compile(
        r"^(PRIMERO|PRIMERA|SEGUNDO|SEGUNDA|TERCERO|TERCERA|CUARTO|CUARTA|"
        r"QUINTO|QUINTA|SEXTO|SEXTA|SÉPTIMO|SÉPTIMA|SEPTIMO|SEPTIMA|OCTAVO|OCTAVA|NOVENO|NOVENA|"
        r"DECIMO|DECIMA|DÉCIMO|DÉCIMA|UNDÉCIMO|UNDÉCIMA|DECIMOPRIMERO|DECIMOPRIMERA|DUODÉCIMO|DUODÉCIMA|DUODECIMA|DUODECIMO|"
        r"DECIMOSEGUNDO|DECIMOSEGUNDA|DECIMOTERCERO|DECIMOTERCERA|"
        r"DECIMOCUARTO|DECIMOCUARTA|DECIMOQUINTO|DECIMOQUINTA|DECIMOSEXTO|DECIMOSEXTA|"
        r"DECIMOSÉPTIMO|DECIMOSÉPTIMA|DECIMOCTAVO|DECIMOCTAVA|DECIMONOVENO|DECIMONOVENA|"
        r"VIGÉSIMO|VIGÉSIMA|VIGESIMOPRIMERO|VIGESIMOPRIMERA|VIGESIMOSEGUNDO|VIGESIMOSEGUNDA|"
        r"VIGESIMOTERCERO|VIGESIMOTERCERA|VIGESIMOCUARTO|VIGESIMOCUARTA|VIGESIMOQUINTO|VIGESIMOQUINTA|"
        r"VIGESIMOSEXTO|VIGESIMOSEXTA|VIGESIMOSÉPTIMO|VIGESIMOSÉPTIMA|VIGESIMOOCTAVO|VIGESIMOOCTAVA|"
        r"VIGESIMONOVENO|VIGESIMONOVENA|TRIGÉSIMO|TRIGÉSIMA)\.?", re.IGNORECASE
    )

    def new_clause(text: str) -> Dict:
        return {
            "text": text.strip(),
            "label_expected": {k: "" for k in ["legality", "abusiveness", "risk_areas", "vagueness", "legal_gaps"]},
            "label_gpt4": {k: "" for k in ["legality", "abusiveness", "risk_areas", "vagueness", "legal_gaps", "alternative_drafting"]},
            "label_DS": {k: "" for k in ["legality", "abusiveness", "risk_areas", "vagueness", "legal_gaps", "alternative_drafting"]},
            "label_GPT4o": {k: "" for k in ["legality", "abusiveness", "risk_areas", "vagueness", "legal_gaps", "alternative_drafting"]},
            "label_GPT35": {k: "" for k in ["legality", "abusiveness", "risk_areas", "vagueness", "legal_gaps", "alternative_drafting"]}
        }

    results = {}

    for filename in os.listdir(directory):
        if filename.lower().endswith(".docx"):
            doc_path = os.path.join(directory, filename)
            doc = Document(doc_path)

            clauses = []
            current_clause = ""

            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                if regex_clause.match(text):
                    if current_clause:
                        clauses.append(new_clause(current_clause))
                    current_clause = text
                else:
                    current_clause += " " + text

            if current_clause:
                clauses.append(new_clause(current_clause))

            results[filename] = clauses

    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)

    print(f"Results saved to '{output_json}'.")

    return results




def copy_labels(
    reference_json_path: str,
    target_json_path: str
) -> int:
    """
    Copies the 'label_expected' values from a reference JSON file and
    overwrites the target file with the updated content.

    **WARNING: This function is destructive and will overwrite the file
    specified in `target_json_path`.**

    Args:
        reference_json_path (str): Path to the JSON file containing the correct labels (the source).
        target_json_path (str): Path to the JSON file to be updated and overwritten.

    Returns:
        int: The number of sections successfully updated. Returns 0 if there is an error.
    """
    print(f"Starting to modify file: '{target_json_path}'...")

    # --- Step 1: Load data from both files into memory ---
    try:
        with open(reference_json_path, "r", encoding="utf-8") as f:
            reference_data = json.load(f)
        with open(target_json_path, "r", encoding="utf-8") as f:
            target_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return 0
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file. Details: {e}")
        return 0

    # --- Step 2: Create a lookup map from the reference file ---
    label_map = {}
    for region, sections in reference_data.items():
        for section in sections:
            if "text" in section and "label_expected" in section:
                key_text = section["text"].strip()
                label_map[key_text] = section["label_expected"]

    print(f"Loaded {len(label_map)} labels from the reference file.")

    # --- Step 3: Modify data in memory ---
    labels_copied = 0
    sections_reviewed = 0
    for region, sections in target_data.items():
        for section in sections:
            sections_reviewed += 1
            if "text" in section:
                key_text_target = section["text"].strip()
                if key_text_target in label_map:
                    section["label_expected"] = label_map[key_text_target]
                    labels_copied += 1

    # --- Step 4: Overwrite the target file with modified data ---
    with open(target_json_path, "w", encoding="utf-8") as f:
        json.dump(target_data, f, indent=4, ensure_ascii=False)

    # --- Final report ---
    print("\n--- Process completed ---")
    print(f"Total sections reviewed: {sections_reviewed}")
    print(f"'label_expected' values copied and updated: {labels_copied}")
    print(f"File '{target_json_path}' has been modified and saved.")

    return labels_copied
