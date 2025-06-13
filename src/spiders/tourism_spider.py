import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from pathlib import Path
import re
from datetime import datetime

class CubaTourismSpider(CrawlSpider):
    name = "cuba_tourism"
    
    # Lista de ciudades principales de Cuba para mejor categorización
    CITIES = [
        'habana', 'santiago', 'varadero', 'trinidad', 'cienfuegos', 
        'viñales', 'vinales', 'holguin', 'holguín', 'cayo coco',
        'guardalavaca', 'baracoa', 'santa clara', 'matanzas', 'pinar del río'
    ]
    
    custom_settings = {
        'FEEDS': {
            str(Path(__file__).parent.parent / 'data' / 'processed' / 'normalized_data.json'): {
                'format': 'json',
                'encoding': 'utf8',
                'overwrite': True
            }
        },
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 2,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'DEPTH_LIMIT': 3  # Limita la profundidad del crawling
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [
            "https://www.cuba.travel/destinos",
            "https://www.hicuba.com/destinos.htm",
            "https://www.viajehotelescuba.com/provincias",
            "https://www.buenviajeacuba.com/informacion-destinos/",
            "https://visitcubago.com/mejores-lugares-turisticos-visitar-en-cuba/",
            "https://www.sitiosturisticos.es/paises-de-america/30-lugares-para-conocer-en-cuba/"
        ]

    # Reglas para seguir enlaces relevantes
    rules = (
        Rule(
            LinkExtractor(
                allow=(r'/(ciudad|destino|lugar|turismo|tourist|destination|city)/.*cuba.*',
                       r'/cuba/.*(ciudad|destino|lugar|turismo).*',
                       r'/(habana|santiago|varadero|trinidad|cienfuegos|vinales|matanzas|isla|cayo|santa clara|holguin|sierra maestra).*',
                       r'/hotel/', r'/vuelos/', r'/forum/', r'/review/')
            ),
            callback='parse',
            follow=True
        ),
    )

    def identify_city(self, text, url):
        """Identifica la ciudad mencionada en el texto o URL"""
        text_lower = text.lower()
        for city in self.CITIES:
            if city in text_lower or city in url.lower():
                return city
        return None

    def parse(self, response):
        # Extraer contenido principal
        content_selectors = [
            'main p::text',
            'article p::text',
            '.content p::text',
            '.description::text',
            '#main-content p::text'
        ]
        
        content = []
        for selector in content_selectors:
            content.extend(response.css(selector).getall())
        
        content = ' '.join(content)
        
        # Extraer título y descripción
        title = response.css('h1::text, title::text').get()
        description = response.css('meta[name="description"]::attr(content)').get()
        
        # Identificar ciudad
        city = self.identify_city(content + ' ' + (title or ''), response.url)
        
        if not content or not city:
            return None
            
        # Extraer atracciones específicas
        attractions = []
        attraction_selectors = [
            '.attraction::text',
            '.place::text',
            '.destination::text',
            'h2::text',
            'h3::text'
        ]
        
        for selector in attraction_selectors:
            items = response.css(selector).getall()
            attractions.extend([item.strip() for item in items if any(city in item.lower() for city in self.CITIES)])
        
        # Filtrar contenido no relevante
        if len(content) < 100:  # Ignorar páginas con poco contenido
            return None
            
        return {
            'city': city,
            'url': response.url,
            'title': title,
            'description': description,
            'content': content,
            'attractions': list(set(attractions)),  # Eliminar duplicados
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'source': response.url,
                'crawl_date': datetime.now().isoformat(),
                'language': response.headers.get('content-language', b'').decode() or 'es'
            }
        }

    def closed(self, reason):
        """Log al terminar el crawling"""
        print(f"Spider cerrado: {reason}")