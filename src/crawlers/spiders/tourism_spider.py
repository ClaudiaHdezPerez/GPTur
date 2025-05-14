import scrapy

class CubaTourismSpider(scrapy.Spider):
    name = "cuba_tourism"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = kwargs.get('start_urls', [
            "https://www.mintur.gob.cu/destinos/"
        ])

    def parse(self, response):
        # Tu lógica de scraping aquí
        yield {
            "url": response.url,
            "content": response.text[:500]  # Ejemplo
        }