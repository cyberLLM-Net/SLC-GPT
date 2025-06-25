

import os
import json
import openai
from typing import Dict, List

def contract_analysis(
    input_json: str,
    output_json: str,
    openai_api_key: str,
    deepseek_api_key: str,
) -> None:
    """
    Analyzes contracts clause by clause from a JSON file using multiple LLMs.
    The detailed analysis with prompt_struct is only performed using GPT-4o.

    Args:
        input_json (str): Path to the input JSON file containing contracts to analyze.
        output_json (str): Path to the output JSON file with results.
        openai_api_key (str): API key for OpenAI (GPT-4, GPT-4o, GPT-3.5).
        deepseek_api_key (str): API key for DeepSeek.
        reports_folder (str): Folder where .txt reports will be saved.
    """

    # Client initialization
    clients = {
        "gpt4": openai.OpenAI(api_key=openai_api_key),
        "GPT4o": openai.OpenAI(api_key=openai_api_key),
        "GPT35": openai.OpenAI(api_key=openai_api_key),
        "DS": openai.OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    }

    models = {
        "gpt4": "gpt-4",
        "GPT4o": "gpt-4o",
        "GPT35": "gpt-3.5-turbo",
        "DS": "deepseek-chat"
    }

    prompt_base = """


    El análisis consiste en, por cada cláusula:

    Actúa como un Abogado, Doctor en Derecho y Profesor de Facultad  de Derecho  ubicado en España, especialista en arrendamientos de viviendas según el derecho español. Con la información que se te va pedir, se preparará un informe de para un cliente institucional español (una SOCIMI)  que es una sociedad anónima cotizada en Bolsa cuya actividad principal es la adquisición, promoción y rehabilitación de activos de naturaleza urbana para su arrendamiento, ya sea directa o indirectamente.

    1. El análisis que se te pedirá deberá basarse en el contrato que se adjunta y tendrá que realizarse cláusula a cláusula, no deberá quedar ninguna cláusula sin examinar.

    2.1. Cláusulas Ilegales. Deberás indicar si la cláusula es ilegal.

    2.2. Cláusulas Abusivas. Deberás indicar si la cláusula es abusiva.

    2.3. Identificar posibles áreas de riesgo.  Deberás indicar si la cláusula tiene áreas de riesgo.

    2.4. Identificar posibles áreas términos vagos. Deberás indicar si la cláusula tiene términos vagos.

    2.5. Deberás verificar si de la legislación vigente contrastada con el contenido de este contrato hay alguna laguna jurídica que podría afectarlo.Ten en cuenta el siguiente concepto de laguna jurídica: Para los juristas prácticos, se dice que existe una laguna jurídica cuando un determinado caso exige una solución jurídica y esta no es posible encontrarla dentro del entramado normativo del ordenamiento jurídico al no existir la norma que contemple tal situación.

    """


    prompt_struct = """
        Vas a recibir un informe jurídico generado por una IA legal sobre una cláusula de contrato. Tu tarea es **extraer 5 etiquetas simples** a partir de ese análisis extenso.

        Responde solo con este formato:

        legality: L o I  
        abusiveness: Y o N  
        risk_areas: Y o N  
        vagueness: Y o N  
        legal_gaps: Y o N

        No incluyas explicaciones, solo la clasificación.
            """



    with open(input_json, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    for region, sections in contracts.items():
        for section in sections:
            text = section.get("text", "").strip()
            if not text:
                continue

            # Run prompt_base and prompt_redaccion with all models
            base_responses = {}
            for model_key, client in clients.items():
                # --- Legal analysis ---
                try:
                    response = client.chat.completions.create(
                        model=models[model_key],
                        messages=[
                            {"role": "system", "content": prompt_base},
                            {"role": "user", "content": text}
                        ],
                        temperature=0.2
                    ).choices[0].message.content
                except Exception as e:
                    response = f"[ERROR: {e}]"
                base_responses[model_key] = response

                # --- Alternative drafting ---
                prompt_redaccion = """
                    Tu tarea es realizar una **redacción alternativa de cada cláusula del contrato de arrendamiento**, manteniendo su contenido legal esencial, pero expresándolo con mayor claridad, precisión jurídica y estilo profesional. El objetivo es ofrecer una versión revisada y jurídicamente sólida de cada cláusula.

                    Actúa como un **Abogado, Doctor en Derecho y Profesor universitario en España**, especializado en contratos de arrendamiento de vivienda conforme al derecho español. La reescritura está destinada a una **SOCIMI española**, una sociedad anónima cotizada especializada en la adquisición, promoción y rehabilitación de activos urbanos para su arrendamiento.

                    Para cada cláusula (y para el preámbulo si existe), realiza únicamente lo siguiente:

                    - **Redacción alternativa**: Reescribe la cláusula con un lenguaje más técnico, preciso y claro. Conserva el contenido legal esencial, pero mejora su redacción y estilo. No añadas análisis ni juicios legales, solo reescribe.

                    Evita alterar el significado jurídico, a menos que el texto original sea claramente ambiguo o incorrecto. Utiliza un tono formal, adecuado para un contrato profesional entre partes institucionales.
                    """

                try:
                    redraft = client.chat.completions.create(
                        model=models[model_key],
                        messages=[
                            {"role": "system", "content": prompt_redaccion},
                            {"role": "user", "content": text}
                        ],
                        temperature=0.2
                    ).choices[0].message.content
                except Exception as e:
                    redraft = f"[ERROR REDACCIÓN: {e}]"

                # Save results to section
                label_key = f"label_{model_key}"
                if label_key not in section:
                    section[label_key] = {}

                section[label_key]["llm_response"] = response
                section[label_key]["alternative_drafting"] = redraft

                # Structured classification with GPT-4o on each model's response
                try:
                    struct_response = clients["GPT4o"].chat.completions.create(
                        model=models["GPT4o"],
                        messages=[
                            {"role": "system", "content": prompt_struct},
                            {"role": "user", "content": response}
                        ],
                        temperature=0.2
                    ).choices[0].message.content
                except Exception as e:
                    struct_response = f"[ERROR EN CLASIFICACIÓN: {e}]"

                labels = {
                    "legality": "",
                    "abusiveness": "",
                    "risk_areas": "",
                    "vagueness": "",
                    "legal_gaps": ""
                }

                for line in struct_response.strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        if key in labels:
                            labels[key] = value.strip()

                if label_key not in section:
                    section[label_key] = {}

                section[label_key].update(labels)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=4, ensure_ascii=False)

    print(f"Contracts analyzed. Saved to '{output_json}'.")

    


import json

def check_discrepancies(contracts):
    for region, sections in contracts.items():
        for section in sections:
            review = []

            expected_label = section.get("label_expected")
            if not expected_label:
                continue  # No reference available for comparison

            # Check if any field in label_expected is empty
            if any(value == "" for value in expected_label.values()):
                continue  # Skip this section if any expected label field is empty

            # Iterate over all keys starting with label_ (except label_expected)
            for label_key, content in section.items():
                if label_key.startswith("label_") and label_key != "label_expected":
                    for field, expected_value in expected_label.items():
                        actual_value = content.get(field)
                        if actual_value != expected_value:
                            review.append(f"{field.capitalize()} mismatch in {label_key}")

            # Save discrepancies if any
            if review:
                section["review"] = review
            elif "review" in section:
                # Remove previous discrepancies if no longer present
                del section["review"]

    return contracts


def run_discrepancy_check(json_path: str, output_path: str = "revised.json"):
    """
    Loads a JSON file, applies discrepancy checking, and saves the result.

    Args:
        json_path (str): Path to the input JSON file.
        output_path (str): Path where the updated JSON file will be saved.
    """
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as file:
        contracts = json.load(file)

    # Apply the check
    updated_contracts = check_discrepancies(contracts)

    # Save the result
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(updated_contracts, file, ensure_ascii=False, indent=4)

    print("Discrepancy check completed.")
