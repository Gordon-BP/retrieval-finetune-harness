from src.ArticleScraper import ArticleScraper

# Goes through the sites in sitemap.xml and 
# does a bit of scraping. Make sure you have
# the site owner's permission to scrape and 
# use their data!

if __name__ == "__main__":
     scraper = ArticleScraper("./data/sitemap.xml")
     scraper(limit=1)
     scraper.retry_failures()