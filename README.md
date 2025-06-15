# GPTur

*¿Deseos de vacacionar en Cuba? Su asesor turístico favorito ya esta aquí!!!*

## Autores
- Claudia Hernández Pérez -> C-312
- Joel Aparicio Tamayo -> C-312
- Kendry Javier Del Pino Barbosa -> C-312 

## Descripción del problema
GPTur es un sistema de agentes inteligentes para la generación y actualización automática de información turística sobre destinos en Cuba. El programa funciona como un chatbot que intenta 
ejercer el rol de asistente turístico, basándose principalmente en cuatro categorías: gastronomía, 
alojamiento, vida nocturna y lugares históricos, aunque puede responder consultas más generales. Además de brindar información, también tendrá la capacidad de planear itinerarios.

## Requerimientos para el correcto desempeño
- Versión de `python` mayor que la 10.0 y menor que la 13.0
- Dependencias listadas en [requirements.txt](requirements.txt)
- Herramienta o consola capaz de ejecutar código de `bash`, para poder ejecutar el programa.

## APIs utilizadas
- Mistral AI (para validación y generación de respuestas)
- Scrapy (para crawling de sitios turísticos)
- Otras APIs de procesamiento de lenguaje natural y almacenamiento vectorial, según configuración en [src/settings.py](src/settings.py)

## Uso y ejecución del proyecto

1. Instala las dependencias:
   ```sh
   pip install -r src/requirements.txt
   ```

2. Ejecutar el siguiente comando en la raíz del proyecto

    ```sh
    ./startup.sh
   ```

---
