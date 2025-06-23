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
            "label_expected": {k: "" for k in ["legalidad", "abusividad", "áreas de riesgo", "vaguedades", "lagunas"]},
            "label_gpt4": {k: "" for k in ["legalidad", "abusividad", "áreas de riesgo", "vaguedades", "lagunas", "redaccion_alternativa"]},
            "label_DS": {k: "" for k in ["legalidad", "abusividad", "áreas de riesgo", "vaguedades", "lagunas", "redaccion_alternativa"]},
            "label_GPT4o": {k: "" for k in ["legalidad", "abusividad", "áreas de riesgo", "vaguedades", "lagunas", "redaccion_alternativa"]},
            "label_GPT35": {k: "" for k in ["legalidad", "abusividad", "áreas de riesgo", "vaguedades", "lagunas", "redaccion_alternativa"]}
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
