#!/bin/bash

JSON_FILE="src/data/processed/normalized_data.json"

if [ ! -s "$JSON_FILE" ]; then
    echo "El archivo JSON no existe o está vacío. Ejecutando Scrapy..."
    scrapy crawl cuba_tourism -o "$JSON_FILE"
    
    # Verificar si el crawler tuvo éxito
    if [ ! -s "$JSON_FILE" ]; then
        echo "Error: El crawler no generó datos. Saliendo..."
        exit 1
    fi
else
    echo "El archivo JSON ya contiene datos. Saltando el crawler."
fi

cd src && streamlit run app.py