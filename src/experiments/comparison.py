import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from chatbot.core import CubaChatbot
from chatbot.gap_detector import GapDetector
from crawlers.dynamic_crawler import DynamicCrawler
from agents.retriever_agent import RetrieverAgent
from agents.generator_agent import GeneratorAgent
from agents.gap_detector_agent import GapDetectorAgent
from agents.updater_agent import UpdaterAgent
from agents.agent_manager import AgentManager
from agents.guide_agent import GuideAgent
from agents.planner_agent import TravelPlannerAgent

# Cargar variables de entorno
load_dotenv()
client = MistralClient(api_key="XEV0fCx3MqiG9HqVkGc4Hy5qyD3WwPHr")  # Inicializar cliente Mistral

# Mover las inicializaciones dentro de las funciones que las necesitan
def get_chatbot():
    if 'chatbot' not in globals():
        global chatbot
        chatbot = CubaChatbot()
        if not chatbot.vector_db.get_documents():
            chatbot.vector_db.reload_data()
            
    return chatbot

# 1. Función para tu chatbot
def your_chatbot_response(question):
    """
    Procesa una pregunta y devuelve la respuesta del chatbot
    :param question: Pregunta del usuario
    :return: Respuesta del chatbot
    """
    chatbot = get_chatbot()

    # Inicializar componentes
    detector = GapDetector(chatbot.vector_db)
    updater = DynamicCrawler()
    guide_agent = GuideAgent(chatbot.vector_db)
    planner_agent = TravelPlannerAgent(chatbot.vector_db)

    # Crear agentes
    retriever_agent = RetrieverAgent(chatbot.vector_db)
    generator_agent = GeneratorAgent(guide_agent, planner_agent)
    gap_detector_agent = GapDetectorAgent(detector)
    updater_agent = UpdaterAgent(updater)

    # Manager de agentes
    manager = AgentManager([
        retriever_agent,
        generator_agent,
        gap_detector_agent,
        updater_agent
    ])
    
    # Recuperar contexto relevante
    retrieval_task = {"type": "retrieve", "query": question}
    context = manager.dispatch(retrieval_task, {})
    
    # Generar respuesta inicial
    generate_task = {"type": "generate", "prompt": question}
    response = manager.dispatch(generate_task, context)
    
    # Convertir respuesta a texto si es necesario
    if hasattr(response, "choices"):
        response_text = " ".join([choice.message.content for choice in response.choices])
    else:
        response_text = str(response)
 
    detect_task = {"type": "detect_gap", "prompt": question, "response": response_text}
    needs_update = manager.dispatch(detect_task, context)
    
    if needs_update:
        print("🔄 Actualizando información...")
        sources = detector.identify_outdated_sources(question)
        update_task = {"type": "update_sources", "sources": sources}
        manager.dispatch(update_task, context)
        chatbot.vector_db.reload_data()
        print("✅ Actualización completada")
        
        # Regenerar respuesta con datos actualizados
        response = manager.dispatch(generate_task, context)
        if hasattr(response, "choices"):
            response_text = " ".join([choice.message.content for choice in response.choices])
        else:
            response_text = str(response)
    
    return response_text

# 2. Chatbot alternativo (ahora usando Mistral AI)
def alternative_chatbot_response(question):
    """Chatbot alternativo especializado en turismo cubano usando Mistral"""
    system_prompt = """
    Eres un asistente de viajes especializado en Cuba. Responde preguntas sobre:
    - Destinos turísticos populares y menos conocidos
    - Recomendaciones de itinerarios
    - Información cultural e histórica
    - Consejos prácticos para viajeros
    """
    
    try:
        response = client.chat(
            model="mistral-large-latest",  # Modelo de Mistral
            messages=[
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=question)
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error en chatbot alternativo: {str(e)}")
        return "Error: No se pudo generar respuesta"
    
# 3. Generación de preguntas turísticas con Mistral
def generate_tourism_questions(n=10, country="Cuba"):
    """Genera preguntas turísticas usando Mistral"""
    prompt = f"""
    Genera {n} preguntas específicas sobre turismo en {country} que cubran:
    - Destinos menos conocidos
    - Rutas de senderismo
    - Recomendaciones gastronómicas locales
    - Eventos culturales
    - Opciones de ecoturismo
    - Transporte entre ciudades
    - Requisitos de viaje
    - Itinerarios personalizados
    
    Formato: Lista numerada
    """
    
    try:
        response = client.chat(
            model="mistral-large-latest",
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.7,
            max_tokens=500
        )
        
        # Parsear la respuesta
        content = response.choices[0].message.content
        questions = []
        for line in content.split("\n"):
            if line.strip() and (line[:2].strip().isdigit() or "?" in line):
                # Eliminar números y puntos iniciales
                clean_line = line.split(". ", 1)[-1] if ". " in line else line
                questions.append(clean_line.strip())
                
        return questions[:n]
    except Exception as e:
        print(f"Error generando preguntas: {str(e)}")
        return [
            "¿Cuáles son las mejores playas para familias en Cuba?",
            "¿Qué documentos necesito para viajar a Cuba desde Colombia?",
            "Recomiéndame un itinerario de 7 días en La Habana",
            "¿Dónde puedo disfrutar de la mejor música cubana auténtica?",
            "¿Es seguro viajar por cuenta propia en Cuba?"
        ][:n]
        
# 4. Evaluación comparativa con Mistral
def evaluate_responses(question, response_a, response_b):
    """Evalúa las respuestas usando Mistral"""
    prompt = f"""
    Evalúa las dos respuestas a la pregunta turística sobre Cuba usando estos criterios:
    1. Precisión de la información (0-3 puntos)
    2. Relevancia para el turista (0-2 puntos)
    3. Utilidad práctica (0-2 puntos)
    4. Riqueza de detalles específicos (0-2 puntos)
    5. Claridad en la presentación (0-1 punto)
    
    Pregunta: {question}
    
    Respuesta A: {response_a}
    Respuesta B: {response_b}
    
    Proporciona el puntaje total para cada respuesta (0-10) en formato JSON:
    {{
        "score_A": 0-10,
        "score_B": 0-10,
        "explanation": "Breve justificación"
    }}
    
    Solo devuelve el objeto JSON, sin texto adicional.
    """
    
    try:
        response = client.chat(
            model="mistral-large-latest",
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=300
        )
        
        # Extraer y limpiar el JSON de la respuesta
        json_str = response.choices[0].message.content
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(json_str)
        return evaluation
    except Exception as e:
        print(f"Error en evaluación: {str(e)}")
        return {"score_A": 5, "score_B": 5, "explanation": "Error en evaluación"}
    
# 5. Análisis estadístico
def analyze_results(results):
    scores_a = [res["score_A"] for res in results]
    scores_b = [res["score_B"] for res in results]
    
    # Estadísticas descriptivas
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    std_a = np.std(scores_a)
    std_b = np.std(scores_b)
    
    # Prueba t pareada
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    
    # Proporción de victorias
    wins_a = sum(1 for i in range(len(scores_a)) if scores_a[i] > scores_b[i])
    wins_b = sum(1 for i in range(len(scores_a)) if scores_b[i] > scores_a[i])
    ties = len(scores_a) - wins_a - wins_b
    
    # Tamaño del efecto (Cohen's d)
    d = (mean_a - mean_b) / np.sqrt((std_a**2 + std_b**2) / 2)
    
    return {
        "mean_score_A": mean_a,
        "mean_score_B": mean_b,
        "std_score_A": std_a,
        "std_score_B": std_b,
        "p_value": p_value,
        "wins_A": wins_a,
        "wins_B": wins_b,
        "ties": ties,
        "effect_size": d
    }
    
# Flujo principal de experimentación
def run_experiment(num_questions=10):
    """Ejecuta el experimento completo"""
    # Generar preguntas
    questions = generate_tourism_questions(num_questions, "Cuba")
    results = []
    
    print(f"\n{'='*50}\nIniciando experimento con {len(questions)} preguntas\n{'='*50}")
    
    for i, question in enumerate(questions):
        print(f"\nPregunta {i+1}/{len(questions)}: {question}")
        
        # Obtener respuestas
        your_response = your_chatbot_response(question)
        alt_response = alternative_chatbot_response(question)
        
        print(f"\nTu respuesta ({len(your_response)} caracteres):\n{your_response[:200]}...")
        print(f"\nRespuesta alternativa ({len(alt_response)} caracteres):\n{alt_response[:200]}...")
        
        # Evaluar
        evaluation = evaluate_responses(question, your_response, alt_response)
        results.append({
            "question": question,
            "your_response": your_response,
            "alternative_response": alt_response,
            **evaluation
        })
        
        print(f"\nEvaluación: Tu chatbot {evaluation['score_A']} vs Alternativo {evaluation['score_B']}")
        print(f"Explicación: {evaluation['explanation']}")
        
        # Esperar para evitar rate limits
        wait_time = 15 if i % 3 == 0 else 5
        print(f"Esperando {wait_time} segundos...")
        time.sleep(wait_time)
    
    # Análisis final
    analysis = analyze_results(results)
    
    print(f"\n{'='*50}\nRESULTADOS FINALES\n{'='*50}")
    print(f"Puntuación promedio Tu Chatbot: {analysis['mean_score_A']:.2f} ± {analysis['std_score_A']:.2f}")
    print(f"Puntuación promedio Alternativo: {analysis['mean_score_B']:.2f} ± {analysis['std_score_B']:.2f}")
    print(f"p-value: {analysis['p_value']:.5f}")
    print(f"Victorias: Tu Chatbot {analysis['wins_A']} - Alternativo {analysis['wins_B']} (Empates: {analysis['ties']})")
    print(f"Tamaño del efecto (Cohen's d): {analysis['effect_size']:.2f}")
    
    # Interpretación
    if analysis['p_value'] < 0.05:
        if analysis['mean_score_A'] > analysis['mean_score_B']:
            print("CONCLUSIÓN: Tu chatbot es significativamente mejor (p < 0.05)")
        else:
            print("CONCLUSIÓN: El chatbot alternativo es significativamente mejor (p < 0.05)")
    else:
        print("CONCLUSIÓN: No hay diferencia significativa entre los chatbots")
    
    # Guardar resultados
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"experiment_results_{timestamp}.csv"
    pd.DataFrame(results).to_csv(filename, index=False)
    print(f"\nResultados guardados en {filename}")
    
    return results, analysis

# Ejecutar experimento
results, analysis = run_experiment(num_questions=10)