import scrapy
from scrapy.exporters import JsonItemExporter
from datetime import datetime
from pathlib import Path

class CubaTourismSpider(scrapy.Spider):
    name = "cuba_tourism"
    
    custom_settings = {
        'FEEDS': {
            str(Path(__file__).parent.parent / 'data' / 'processed' / 'normalized_data.json'): {
                'format': 'json',
                'encoding': 'utf8',
                'overwrite': True
            }
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = kwargs.get('start_urls', [
            "https://www.lonelyplanet.com/cuba",
            "https://www.tripadvisor.com/Tourism-g147270-Cuba-Vacations.html",
            "https://www.cuba.travel",
            "https://www.hicuba.com/destinos.htm",
            "https://www.viajehotelescuba.com/provincias",
            "https://www.buenviajeacuba.com/informacion-destinos/",
            "https://visitcubago.com/mejores-lugares-turisticos-visitar-en-cuba/",
            "https://www.sitiosturisticos.es/paises-de-america/30-lugares-para-conocer-en-cuba/"
        ])

    def parse(self, response):
        content = ' '.join(response.css('main p::text, article p::text').getall())
        url = response.url
        title = response.css('title::text').get() or response.url
        
        if content and url and title:
            yield {
                'page_content': content,
                'metadata': {
                    'source': url,
                    'title': title
                }
            }

class CustomPipeline:
    def open_spider(self, spider):
        self.file = open('../data/processed/normalized_data.json', 'w')
        self.exporter = JsonItemExporter(self.file)
        self.exporter.start_exporting()

    def process_item(self, item, spider):
        # Estructura requerida por el JSONLoader
        normalized_item = {
            "page_content": item["page_content"],
            "metadata": item["metadata"]
        }
        self.exporter.export_item(normalized_item)
        return item

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()