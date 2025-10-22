import os
import json
from typing import Dict, List, Optional

import openai                           

import google.generativeai as genai     


def _init_openai_client(api_key: str, base_url: Optional[str] = None):
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url)
    return openai.OpenAI(api_key=api_key)


def _init_gemini_client(api_key: str):
    genai.configure(api_key=api_key)
    return genai


def call_llm(
    provider: str,
    client,
    model: str,
    system_prompt: str,
    user_text: str,
) -> str:
    """
    Adaptador multi-proveedor.
    - OpenAI/DeepSeek por defecto: chat.completions.create
    - Gemini: generate_content
    """
    try:
        if provider in ("openai",) and model.startswith("gpt-5"):
            prompt = f"{system_prompt.strip()}\n\n---\n\n{user_text.strip()}"
            resp = client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": "high"},
                text={"verbosity": "high"},
            )
            return getattr(resp, "output_text", "") or ""

        if provider in ("openai", "openai_compat"):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]
            )
            return resp.choices[0].message.content or ""

        elif provider == "gemini":
            prompt = f"{system_prompt.strip()}\n\n---\n\n{user_text.strip()}"
            model_obj = client.GenerativeModel(model_name=model)
            resp = model_obj.generate_content(prompt)
            return getattr(resp, "text", "") or ""

        else:
            return f"[ERROR: Proveedor no soportado: {provider}]"

    except Exception as e:
        return f"[ERROR: {e}]"



def contract_analysis(
    input_json: str,
    output_json: str,
    openai_api_key: str,
    deepseek_api_key: str,
    gemini_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> None:
    """
    Analiza cláusulas de contratos desde un JSON usando múltiples LLMs.
    Añadidos: Gemini (Google) y Claude (Anthropic).

    Args:
        input_json (str): ruta del JSON de entrada.
        output_json (str): ruta del JSON de salida.
        openai_api_key (str): API key de OpenAI.
        deepseek_api_key (str): API key de DeepSeek.
        gemini_api_key (str, opcional): API key de Google Gemini.
        anthropic_api_key (str, opcional): API key de Anthropic (Claude).
    """

    clients = {
        # OpenAI "nativo"
        "gpt4": {"provider": "openai", "client": _init_openai_client(openai_api_key)},
        "GPT4o": {"provider": "openai", "client": _init_openai_client(openai_api_key)},
        "GPT35": {"provider": "openai", "client": _init_openai_client(openai_api_key)},
        "GPT5": {"provider": "openai", "client": _init_openai_client(openai_api_key)},
        # DeepSeek (OpenAI-compatible con base_url)
        "DS": {
            "provider": "openai_compat",
            "client": _init_openai_client(deepseek_api_key, base_url="https://api.deepseek.com"),
        },
    }


    if gemini_api_key:
        clients["GEMINI"] = {
            "provider": "gemini",
            "client": _init_gemini_client(gemini_api_key),
        }

    models = {
        "gpt4": "gpt-4",
        "GPT4o": "gpt-4o",
        "GPT35": "gpt-3.5-turbo",
        "DS": "deepseek-chat",
        "GPT5": "gpt-5", 
        "GEMINI": "models/gemini-2.5-pro",        
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

    # Carga de contratos
    with open(input_json, "r", encoding="utf-8") as f:
        contracts = json.load(f)

    for region, sections in contracts.items():
        for section in sections:
            text = section.get("text", "").strip()
            if not text:
                continue

            base_responses = {}

            # análisis y redacción alternativa con todos los modelos configurados 
            for model_key, bundle in clients.items():
                provider = bundle["provider"]
                client = bundle["client"]
                model_name = models[model_key]

                # análisis legal
                response = call_llm(
                    provider=provider,
                    client=client,
                    model=model_name,
                    system_prompt=prompt_base,
                    user_text=text,
                )
                base_responses[model_key] = response

                # redacción alternativa
                prompt_redaccion = """
                    Tu tarea es realizar una **redacción alternativa de cada cláusula del contrato de arrendamiento**, manteniendo su contenido legal esencial, pero expresándolo con mayor claridad, precisión jurídica y estilo profesional. El objetivo es ofrecer una versión revisada y jurídicamente sólida de cada cláusula.

                    Actúa como un **Abogado, Doctor en Derecho y Profesor universitario en España**, especializado en contratos de arrendamiento de vivienda conforme al derecho español. La reescritura está destinada a una **SOCIMI española**, una sociedad anónima cotizada especializada en la adquisición, promoción y rehabilitación de activos urbanos para su arrendamiento.

                    Para cada cláusula (y para el preámbulo si existe), realiza únicamente lo siguiente:

                    - **Redacción alternativa**: Reescribe la cláusula con un lenguaje más técnico, preciso y claro. Conserva el contenido legal esencial, pero mejora su redacción y estilo. No añadas análisis ni juicios legales, solo reescribe.

                    Evita alterar el significado jurídico, a menos que el texto original sea claramente ambiguo o incorrecto. Utiliza un tono formal, adecuado para un contrato profesional entre partes institucionales.
                """

                redraft = call_llm(
                    provider=provider,
                    client=client,
                    model=model_name,
                    system_prompt=prompt_redaccion,
                    user_text=text,
                )

                # Guardado en la sección
                label_key = f"label_{model_key}"
                if label_key not in section:
                    section[label_key] = {}

                section[label_key]["llm_response"] = response
                section[label_key]["alternative_drafting"] = redraft

                # clasificación estructurada con GPT-4o 
                struct_response = call_llm(
                    provider=clients["GPT4o"]["provider"],
                    client=clients["GPT4o"]["client"],
                    model=models["GPT4o"],
                    system_prompt=prompt_struct,
                    user_text=response,
                )

                labels = {
                    "legality": "",
                    "abusiveness": "",
                    "risk_areas": "",
                    "vagueness": "",
                    "legal_gaps": ""
                }

                for line in (struct_response or "").strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        if key in labels:
                            labels[key] = value.strip()

                section[label_key].update(labels)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=4, ensure_ascii=False)

    print(f"Contracts analyzed. Saved to '{output_json}'.")


# utilidades de discrepancias 
def check_discrepancies(contracts):
    for region, sections in contracts.items():
        for section in sections:
            review = []

            expected_label = section.get("label_expected")
            if not expected_label:
                continue  

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
