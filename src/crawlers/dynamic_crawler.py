# dynamic_crawler.py
from apscheduler.schedulers.background import BackgroundScheduler
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor
from spiders.tourism_spider import CubaTourismSpider  # Archivo con tu spider
import threading

class DynamicCrawler:
    def __init__(self):
        self.runner = CrawlerRunner()
        self.scheduler = BackgroundScheduler()
        self.spider = CubaTourismSpider  # Clase spider importada
        
    def update_sources(self, urls):
        """Ejecuta crawling para URLs específicas"""
        self.runner.crawl(self.spider, start_urls=urls)
        
    def crawl_job(self):
        """Tarea programada para crawling completo"""
        self.runner.crawl(self.spider)
        
    def start_scheduler(self):
        """Inicia el planificador en un hilo separado"""
        self.scheduler.add_job(self.crawl_job, 'interval', hours=24)
        self.scheduler.start()
        
        # Ejecutar el reactor de Twisted en hilo secundario
        threading.Thread(target=reactor.run, args=(False,)).start()

    def stop(self):
        """Detener todos los procesos"""
        self.scheduler.shutdown()
        reactor.stop()