from src.ArticleScraper import ArticleScraper

if __name__ == "__main__":
     scraper = ArticleScraper("./data/sitemap.xml")
     scraper(limit=1)
     scraper.retry_failures()