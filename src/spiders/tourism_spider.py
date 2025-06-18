import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from pathlib import Path
import re
from datetime import datetime

class CubaTourismSpider(CrawlSpider):
    name = "cuba_tourism"
    
    CITIES = [
        'habana', 'santiago', 'varadero', 'trinidad', 'cienfuegos', 
        'viñales', 'vinales', 'holguin', 'holguín', 'cayo coco',
        'guardalavaca', 'baracoa', 'santa clara', 'matanzas', 'pinar del río',
        'artemisa', 'mayabeque', 'las tunas', 'camagüey', 'sancti spiritus', 
        'ciego de avila', 'granma', 'sierra maestra', 'guantanamo', 'isla de la juventud',
        'cayo largo', 'cayo santa maria', 'cayo guillermo'
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
        'DOWNLOAD_DELAY': 3,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'DEPTH_LIMIT': 4
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    rules = (
        Rule(
            LinkExtractor(
                allow=(
                    r'/(ciudad|destino|lugar|turismo|tourist|destination|city)/.*cuba.*',
                    r'/cuba/.*(ciudad|destino|lugar|turismo).*',
                    r'/(habana|santiago|varadero|trinidad|cienfuegos|vinales|matanzas|isla|cayo|santa clara|holguin|sierra maestra).*',
                    r'/hotel/', r'/vuelos/', r'/forum/', r'/review/',
                    r'/(playa|excursion|museo|cultura|restaurante|atraccion|guia|travel)/'
                )
            ),
            callback='parse',
            follow=True
        ),
    )

    def identify_city(self, text, url):
        """
        Identify the Cuban city mentioned in the text or URL.

        Args:
            text (str): The text content to analyze
            url (str): The URL of the page being processed

        Returns:
            str: Name of the identified city or None if no city is found
        """
        text_lower = text.lower()
        for city in self.CITIES:
            if city in text_lower or city in url.lower():
                return city
        return None

    def parse(self, response):
        """
        Parse the webpage content and extract relevant tourism information.

        Args:
            response (scrapy.http.Response): The response object containing the webpage

        Returns:
            dict: Structured data including city, content, attractions, and metadata,
                 or None if insufficient content is found
        """
        content_selectors = [
            'main p::text',
            'article p::text',
            '.content p::text',
            '.description::text',
            '#main-content p::text',
            'div.entry-content p::text',
            'section p::text'
        ]
        
        content = []
        for selector in content_selectors:
            content.extend(response.css(selector).getall())
        
        content = ' '.join(content)
        
        title = response.css('h1::text, title::text').get()
        description = response.css('meta[name="description"]::attr(content)').get()
        
        city = self.identify_city(content + ' ' + (title or ''), response.url)
        
        if not content or not city:
            return None
        
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
        
        if len(content) < 100:
            return None
            
        return {
            'city': city,
            'url': response.url,
            'title': title,
            'description': description,
            'content': content,
            'attractions': list(set(attractions)), 
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'source': response.url,
                'crawl_date': datetime.now().isoformat(),
                'language': response.headers.get('content-language', b'').decode() or 'es'
            }
        }

    def closed(self, reason):
        """
        Handle spider closure and perform cleanup operations.

        Args:
            reason (str): The reason for spider closure
        """
        print(f"Spider cerrado: {reason}")