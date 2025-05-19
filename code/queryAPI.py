! pip install openai
import openai
import json
import os
import deepseek

prompt_base = """
El análisis consiste en, por cada cláusula:


Act as a Lawyer, Doctor en Derecho y Profesor de Facultad  de Derecho  ubicado en España, especialista en arrendamientos de viviendas según el derecho español. Con la información que se te va pedir, se preparará un informe de para un cliente institucional español (una SOCIMI)  que es una sociedad anónima cotizada en Bolsa cuya actividad principal es la adquisición, promoción y rehabilitación de activos de naturaleza urbana para su arrendamiento, ya sea directa o indirectamente.


1. El análisis que se te pedirá deberá basarse en el contrato que se adjunta y tendrá que realizarse cláusula a cláusula, no deberá quedar ninguna cláusula sin examinar.


2.1. Cláusulas Ilegales. You must indicate whether the clause es ilegal.


2.2. Cláusulas Abusivas. You must indicate whether the clause es abusiva.


2.3. Identificar posibles áreas de riesgo.  You must indicate whether the clause tiene áreas de riesgo.


2.4. Identificar posibles áreas términos vagos. You must indicate whether the clause tiene términos vagos.


2.5. Deberás verificar si de la legislación vigente contrastada con el contenido de este contrato hay alguna legal loophole que podría afectarlo.Ten en cuenta el siguiente concepto de legal loophole: Para los juristas prácticos, se dice que existe una legal loophole cuando un determinado caso exige una solución jurídica y esta no es posible encontrarla dentro del entramado normativo del ordenamiento jurídico al no existir la norma que contemple tal situación.


"""

prompt_struct = """
Analiza e interpreta el texto enviado para generar un informe con una estructura concreta.
Una respuesta de ejemplo es la siguiente, usa L para marcar legales, I para ilegales, y para las otras 4 columnas N-No; S-Sí:
Legalidad: L 
Abusividad: N
Áreas de riesgo: N
Vaguedades: S
Lagunas: S


"""


# Configuración de clientes
client = openai.OpenAI(api_key="***", base_url="https://api.deepseek.com")    
clientGPT = openai.OpenAI(api_key="***")



def get_new_filename(region, folder):
    region_safe = region.lower().replace(" ", "_")
    files = os.listdir(folder) if os.path.exists(folder) else []
    indices = [
        int(f.split("_")[-1].replace(".txt", "")) 
        for f in files 
        if f.startswith(f"informe_{region_safe}_") and f.endswith(".txt") and f.split("_")[-1].replace(".txt", "").isdigit()
    ]
    new_index = max(indices) + 1 if indices else 1
    return f"informe_{region_safe}_{new_index}.txt"



def analizar_contrato(region, secciones):
    os.makedirs("informes/informesGPT4", exist_ok=True)
    filename = get_new_filename(region, "informes/informesGPT4")


    with open(os.path.join("informes/informesGPT4", filename), "w", encoding="utf-8") as file:
        file.write(f"Contrato de {region}\n\n")


        for seccion in secciones:
            if "text" not in seccion:
                continue


            texto_seccion = seccion["text"]


            # Primera llamada con prompt_base
            mensajes_base = [
                {"role": "system", "content": prompt_base},
                {"role": "user", "content": texto_seccion}
            ]




            respuesta_base = clientGPT.chat.completions.create(
                model="gpt-4",
                messages=mensajes_base,
                temperature=0.2
                ).choices[0].message.content
#            respuesta_base = client.chat.completions.create(
#                model="deepseek-chat",
#                messages=mensajes_base,
#                temperature=0.2
#            ).choices[0].message.content


    #        seccion["label_GPT4o"] = respuesta_base


            # Segunda llamada con prompt_struct
            mensajes_struct = [
                {"role": "system", "content": prompt_struct},
                {"role": "user", "content": respuesta_base}
            ]


            respuesta_struct = clientGPT.chat.completions.create(
                model="gpt-4o",
                messages=mensajes_struct,
                temperature=0.2
            ).choices[0].message.content


# Recuperar las etiquetas antiguas si existen
            etiquetas_antiguas = seccion.get("label_gpt4", {})


            # Procesar la respuesta estructurada
            lineas = respuesta_struct.strip().split("\n")
            etiquetas_nuevas = {
                "legalidad": "", 
                "abusividad": "", 
                "áreas de riesgo": "", 
                "vaguedades": "", 
                "lagunas": ""
            }


            for linea in lineas:
                partes = linea.split(": ")
                if len(partes) == 2:
                    clave = partes[0].strip().lower()
                    valor = partes[1].strip()
                    if clave in etiquetas_nuevas:
                        etiquetas_nuevas[clave] = valor


            # Añadir la respuesta base como parte de las etiquetas
            etiquetas_nuevas["respuesta_llm"] = respuesta_base


            # Actualizar las etiquetas antiguas con las nuevas (sin perder columnas antiguas)
            etiquetas_antiguas.update(etiquetas_nuevas)


            # Guardar en label_GPT4o manteniendo las columnas que no se reescriben
            seccion["label_gpt4"] = etiquetas_antiguas




            # Guardar en archivo de texto
            file.write(f"Sección:\n{texto_seccion}\n\nRespuesta Base:\n{respuesta_base}\n\nRespuesta Struct:\n{respuesta_struct}\n\n{'-'*80}\n\n")


    return secciones



def guardar_json(contratos):
    with open("summary.json", "w", encoding="utf-8") as file:
        json.dump(contratos, file, ensure_ascii=False, indent=4)


# Cargar el JSON con los contratos
with open("summary.json", "r", encoding="utf-8") as file:
    contratos = json.load(file)


# Generar informes y actualizar el JSON
for region, secciones in contratos.items():
    contratos[region] = analizar_contrato(region, secciones)


guardar_json(contratos)
print("Informes generados y guardados en la carpeta informes/informesDS y en el archivo JSON.")

DeepSeek = "informes/informesDS"
GPT35 = "informes/informesGPT35"
GPT4 = "informes/informesGPT4"
GPT4o = "informes/informesGPT4o"


prompt_alternativo= """
IMPORTANTE: Analizarás exclusivamente las cláusulas que comienzan con números ordinales (por ejemplo: Primero, Primera, Segundo, Segunda, etc.). Algunos contratos pueden reiniciar el contador de cláusulas o saltarse cláusulas, por lo que deberás seguir el mismo orden y numeración que figure en el contrato (por ejemplo: Primero, Segundo, Primero, Segundo, Tercero, etc.).


Tu tarea es realizar un **análisis jurídico exhaustivo cláusula por cláusula** de un contrato de arrendamiento, separando cada análisis mediante la palabra clave `###`. También deberás realizar un análisis independiente del contenido previo a la primera cláusula, tratándolo como si fuese una cláusula más.


Actúa como un **Abogado, Doctor en Derecho y Profesor universitario en España**, especialista en contratos de arrendamiento de vivienda conforme al derecho español. El informe está destinado a una **SOCIMI española**, una sociedad anónima cotizada especializada en la adquisición, promoción y rehabilitación de activos urbanos para su arrendamiento.


Para **cada cláusula** (incluyendo el preámbulo), deberás proporcionar las siguientes secciones claramente separadas:


1. **Texto de la cláusula** (copiado literalmente si está disponible).
2. **Resumen jurídico**: Describe brevemente el contenido de la cláusula.
3. **Legalidad**: Indica si la cláusula es ilegal, citando normativa aplicable.
4. **Abusividad**: Indica si la cláusula puede considerarse abusiva según la legislación española.
5. **Áreas de riesgo**: Expón los posibles riesgos legales o contractuales asociados.
6. **Términos vagos**: Señala los términos que puedan generar ambigüedad o interpretación variable.
7. **Lagunas jurídicas**: Valora si existe una legal loophole aplicable conforme a este criterio: 
   - Existe una laguna cuando una situación exige una solución jurídica pero no existe norma que la contemple en el ordenamiento vigente.


Estructura cada análisis con claridad y evita repeticiones innecesarias. Sé técnico y preciso, usando un lenguaje legal estandarizado.


"""




DeepSeek = "informes_cambio_prompt/informesDS_prompt"
GPT35 = "informes_cambio_prompt/informesGPT35_prompt"
GPT4 = "informes_cambio_prompt/informesGPT4_prompt"
GPT4o = "informes_cambio_prompt/informesGPT4o_prompt"


# Función para generar un nombre único para cada informe de una región

def get_new_filename(region, folder, tipo):
    region_safe = region.lower().replace(" ", "_")
    files = os.listdir(folder) if os.path.exists(folder) else []
    indices = []
    for f in files:
        if f.startswith(f"informe_{region_safe}_{tipo}_") and f.endswith(".txt"):
            try:
                index = int(f.split("_")[-1].replace(".txt", ""))
                indices.append(index)
            except ValueError:
                pass
    new_index = max(indices) + 1 if indices else 1
    return f"informe_{region_safe}_{tipo}_{new_index}.txt"



def analizar_contrato(region, secciones):
    # Unir todas las secciones del contrato
    texto_contrato = "\n\n".join(seccion["text"] for seccion in secciones if "text" in seccion)


    client = openai.OpenAI(api_key="***")

   


    # Solicitud con prompt_base
    mensajes_base = [
        {"role": "system", "content": prompt_base},
        {"role": "user", "content": f"Contrato de {region}:\n\n{texto_contrato}"}
    ]
    respuesta_base = client.chat.completions.create(
        model="gpt-4o",
        messages=mensajes_base,
        temperature=0.2
    )
    respuesta_completa_base = respuesta_base.choices[0].message.content


    # Solicitud con prompt_alternativo
    mensajes_alternativo = [
        {"role": "system", "content": prompt_alternativo},
        {"role": "user", "content": f"Contrato de {region}:\n\n{texto_contrato}"}
    ]
    respuesta_alternativa = client.chat.completions.create(
        model="gpt-4o",
        messages=mensajes_alternativo,
        temperature=0.2
    )
    respuesta_completa_alternativa = respuesta_alternativa.choices[0].message.content


    # Guardar ambos archivos
    os.makedirs(GPT4o , exist_ok=True)


    filename_base = get_new_filename(region, GPT4o , "base")
    with open(os.path.join(GPT4o , filename_base), "w", encoding="utf-8") as file:
        file.write(f"Contrato de {region}\n\n{respuesta_completa_base}")


    filename_alternativo = get_new_filename(region, GPT4o , "alternativo")
    with open(os.path.join(GPT4o , filename_alternativo), "w", encoding="utf-8") as file:
        file.write(f"Contrato de {region}\n\n{respuesta_completa_alternativa}")


# Cargar el JSON con los contratos

import json
with open("contratos_labeledDS_2.json", "r", encoding="utf-8") as file:
    contratos = json.load(file)


# Generar informes
for region, secciones in contratos.items():
    analizar_contrato(region, secciones)


print("Informes generados y guardados en la carpeta /informesDS.")




prompt_redaccion = """


Tu tarea es realizar una **redacción alternativa de cada cláusula del contrato de arrendamiento**, manteniendo su contenido legal esencial, pero expresándolo con mayor claridad, precisión jurídica y estilo profesional. El objetivo es ofrecer una versión revisada y jurídicamente sólida de cada cláusula.


Actúa como un **Abogado, Doctor en Derecho y Profesor universitario en España**, especializado en contratos de arrendamiento de vivienda conforme al derecho español. La reescritura está destinada a una **SOCIMI española**, una sociedad anónima cotizada especializada en la adquisición, promoción y rehabilitación de activos urbanos para su arrendamiento.


Para cada cláusula (y para el preámbulo si existe), realiza únicamente lo siguiente:


- **Redacción alternativa**: Reescribe la cláusula con un lenguaje más técnico, preciso y claro. Conserva el contenido legal esencial, pero mejora su redacción y estilo. No añadas análisis ni juicios legales, solo reescribe.


Evita alterar el significado jurídico, a menos que el texto original sea claramente ambiguo o incorrecto. Utiliza un tono formal, adecuado para un contrato profesional entre partes institucionales.


"""



import json

import time

import openai

from pathlib import Path




# Inicialización de clientes
clientes = {
    "gpt4": openai.OpenAI(api_key="***"),
    "GPT4o": openai.OpenAI(api_key="***"),
    "GPT35": openai.OpenAI(api_key="***"),
    "DS": openai.OpenAI(api_key="***", base_url="https://api.deepseek.com")
}


# Mapeo de modelo -> nombre de modelo de la API
modelos_api = {
    "gpt4": "gpt-4",
    "GPT4o": "gpt-4o",
    "GPT35": "gpt-3.5-turbo",
    "DS": "deepseek-chat"
}



def redacciones_por_modelo(json_input_path, json_output_path, sleep_seconds=2):
    with open(json_input_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    for doc, clausulas in data.items():
        print(f"\n📄 Documento: {doc}")
        for idx, clausula in enumerate(clausulas):
            texto_clausula = clausula.get("text", "")


            for modelo in ["gpt4", "GPT4o", "GPT35", "DS"]:
                key = f"label_{modelo}"
                if key not in clausula or not isinstance(clausula[key], dict):
                    continue  # Saltar si no hay datos del modelo


                if "redaccion_alternativa" in clausula[key]:
                    continue  # Ya existe, nos lo saltamos


                mensajes = [
                    {"role": "system", "content": prompt_redaccion},
                    {"role": "user", "content": texto_clausula}
                ]


                try:
                    client = clientes[modelo]
                    model_name = modelos_api[modelo]
                    respuesta = client.chat.completions.create(
                        model=model_name,
                        messages=mensajes,
                        temperature=0.2
                    )
                    redaccion = respuesta.choices[0].message.content.strip()
                    clausula[key]["redaccion_alternativa"] = redaccion
                    print(f"[✓] {modelo} cláusula {idx + 1}")
                except Exception as e:
                    print(f"[✗] Error con {modelo} cláusula {idx + 1}: {e}")
                    clausula[key]["redaccion_alternativa"] = ""


                time.sleep(sleep_seconds)


    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    print(f"\n✅ Guardado en: {json_output_path}")


redacciones_por_modelo(
    json_input_path="summary.json",
    json_output_path="final.json",
    sleep_seconds=1.5
)



def limpiar_labels_sin_expected(contratos):
    for region, secciones in contratos.items():
        for seccion in secciones:
            if "label_expected" not in seccion:
                # Crear una lista de claves a eliminar
                claves_a_eliminar = [clave for clave in seccion if clave.startswith("label_")]
                for clave in claves_a_eliminar:
                    del seccion[clave]
    return contratos


# Cargar el JSON
with open("summaryNew.json", "r", encoding="utf-8") as file:
    contratos = json.load(file)


# Limpiar labels no deseados
contratos = limpiar_labels_sin_expected(contratos)


# Guardar el JSON limpio
with open("summaryNew_Limpio.json", "w", encoding="utf-8") as file:
    json.dump(contratos, file, ensure_ascii=False, indent=4)


print("Limpieza completada. Archivo guardado como summaryGPT4o_limpio.json.")





def verificar_discrepancias(contratos):
    for region, secciones in contratos.items():
        for seccion in secciones:
            revisar = []


            label_expected = seccion.get("label_expected")
            if not label_expected:
                continue  # No hay referencia para comparar


            # Recorremos todas las claves que empiezan con label_ (excepto label_expected)
            for clave_label, contenido in seccion.items():
                if clave_label.startswith("label_") and clave_label != "label_expected":
                    for campo, valor_esperado in label_expected.items():
                        valor_actual = contenido.get(campo)
                        if valor_actual != valor_esperado:
                            revisar.append(f"{campo.capitalize()} no coincide en {clave_label}")


            # Guardar las discrepancias si existen
            if revisar:
                seccion["revisar"] = revisar
            elif "revisar" in seccion:
                # Si antes había y ahora no hay discrepancias, la eliminamos
                del seccion["revisar"]


    return contratos


# Cargar JSON
with open("final.json", "r", encoding="utf-8") as file:
    contratos = json.load(file)


# Aplicar la verificación de discrepancias
contratos = verificar_discrepancias(contratos)


# Guardar el resultado en un nuevo archivo
with open("revisar.json", "w", encoding="utf-8") as file:
    json.dump(contratos, file, ensure_ascii=False, indent=4)


print("Verificación de discrepancias completada.")





import json

from collections import OrderedDict



def asegurar_label_expected_despues_de_text(contratos):
    valores_vacios = {
        "legalidad": "",
        "abusividad": "",
        "áreas de riesgo": "",
        "vaguedades": "",
        "lagunas": ""
    }


    for region, secciones in contratos.items():
        for idx, seccion in enumerate(secciones):
            if "label_expected" not in seccion:
                seccion["label_expected"] = valores_vacios.copy()


            # Reconstruir el diccionario con el orden deseado
            nuevo_orden = []
            for k, v in seccion.items():
                nuevo_orden.append((k, v))
                if k == "text" and "label_expected" in seccion:
                    nuevo_orden.append(("label_expected", seccion["label_expected"]))


            # Si 'text' no estaba, poner 'label_expected' al principio
            if not any(k == "text" for k in seccion):
                nuevo_orden = [("label_expected", seccion["label_expected"])] + [
                    (k, v) for k, v in seccion.items() if k != "label_expected"
                ]


            # Eliminar duplicados en caso de que 'label_expected' ya esté
            seen = set()
            nuevo_orden_unico = []
            for k, v in nuevo_orden:
                if k not in seen:
                    nuevo_orden_unico.append((k, v))
                    seen.add(k)


            secciones[idx] = OrderedDict(nuevo_orden_unico)


    return contratos


# Cargar el JSON
with open("revisar.json", "r", encoding="utf-8") as file:
    contratos = json.load(file)


# Asegurar que label_expected existe y queda justo después de 'text'
contratos = asegurar_label_expected_despues_de_text(contratos)


# Guardar el JSON actualizado
with open("revisar.json", "w", encoding="utf-8") as file:
    json.dump(contratos, file, ensure_ascii=False, indent=4)



