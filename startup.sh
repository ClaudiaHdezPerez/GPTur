#!/bin/bash

scrapy crawl cuba_tourism -o src/data/processed/normalized_data.json

cd src && streamlit run app.py