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
        """
        Execute crawling operations for specific URLs.

        Args:
            urls (list): List of dictionaries containing URLs to crawl

        Note:
            Starts the reactor in a separate thread if not already running
        """
        with self.lock:
            urls = [url['url'] for url in urls]
            d = self.runner.crawl(self.spider, start_urls=urls)
            d.addCallback(self._crawl_callback)
            
            if not reactor.running:
                threading.Thread(target=reactor.run, 
                                args=(False,)).start()

    def _crawl_callback(self, result):
        """
        Callback function executed when crawling is completed.

        Args:
            result: The crawling operation result

        Returns:
            The crawling operation result
        """
        print("Crawling completado. Datos actualizados.")
        return result