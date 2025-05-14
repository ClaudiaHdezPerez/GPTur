import scrapy
from scrapy.crawler import CrawlerProcess

class CubaTourismSpider(scrapy.Spider):
    name = "cuba_tourism"
    custom_settings = {
        'FEED_FORMAT': 'json',
        'FEED_URI': '../data/raw/destinations.json'
    }
    
    start_urls = [
        "https://www.mintur.gob.cu/destinos/",
        "https://www.ecured.cu/Patrimonios_de_la_Humanidad_en_Cuba"
    ]

    def parse(self, response):
        yield {
            'title': response.css('h1::text').get(),
            'content': ' '.join(response.css('div.content p::text').getall()),
            'url': response.url,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(CubaTourismSpider)
    process.start()