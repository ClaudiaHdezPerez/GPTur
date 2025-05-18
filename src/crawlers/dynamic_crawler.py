from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor
from spiders.tourism_spider import CubaTourismSpider
import threading

class DynamicCrawler:
    def __init__(self):
        self.runner = CrawlerRunner()
        self.lock = threading.Lock()
        self.spider = CubaTourismSpider

    def update_sources(self, urls):
        """Ejecuta crawling para URLs específicas"""
        with self.lock:
            d = self.runner.crawl(self.spider, start_urls=urls)
            d.addCallback(self._crawl_callback)
            
            if not reactor.running:
                threading.Thread(target=reactor.run, 
                                args=(False,)).start()

    def _crawl_callback(self, result):
        print("Crawling completado. Datos actualizados.")
        return result