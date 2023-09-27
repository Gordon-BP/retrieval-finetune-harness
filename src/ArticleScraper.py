import asyncio
import aiohttp
import re
import tqdm
import pandas as pd
from typing import Dict, List
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET


class ArticleScraper:
    """
    A class for scraping articles.
    """

    def __init__(self, sitemap_file: str):
        """
        Initialize the scraper with the sitemap file.
        """
        self.sitemap_file = sitemap_file
        self.languages = set()
        self.article_ids = set()

    def get_lang_code(self, url: str) -> str:
        """
        Extract the language code from a URL. It's a two char code like 'hi' for hindi
        English is en-us though
        """
        match = re.search(r"/hc/(.*)/articles", url)
        return match.group(1) if match is not None else None

    def get_article_code(self, url: str) -> str:
        """
        Extract the article ID from a URL.
        """
        match = re.search(r"articles/(\d+)", url)
        return match.group(1) if match is not None else None

    async def fetch_article(
        self, session: aiohttp.ClientSession, lang: str, article_id: int
    ) -> Dict:
        """
        Asynchronously fetch an article given its language and ID.
        """
        # Should probably find a way not to hard code this but eh
        url = f"shhhhhhh-secret/{lang}/articles/{article_id}"
        async with session.get(url) as response:
            assert response.status == 200
            return await response.json()

    async def gather_data(
        self, languages: List[str], article_ids: List[int], limit: int
    ):
        """
        Gather and save article data.
        """
        rate_limit = asyncio.Semaphore(limit)  # limit requests per second
        rows = []
        errors = []
        async with aiohttp.ClientSession() as session:
            tasks = []

            for future in tqdm(
                asyncio.as_completed(tasks), total=len(tasks), desc="Fetching Articles"
            ):
                try:
                    await future
                except Exception as e:
                    print(f"An exception occurred: {e}")

        df = pd.DataFrame.from_records(rows)
        df.to_csv("articles_data.csv", index=False)

        df_errors = pd.DataFrame.from_records(errors)
        df_errors.to_csv("failed_articles.csv")

    async def fetch_and_save(self, session, rate_limit, lang, article_id, rows, errors):
        """
        Asynchronously fetch and save an article.
        """
        async with rate_limit:
            try:
                article_data = await self.fetch_article(session, lang, article_id)
                if "body" in article_data["article"]:
                    soup = BeautifulSoup(article_data["article"]["body"], "lxml")
                    text_content = soup.get_text()
                    article_data["article"]["body"] = text_content
                    rows.append(
                        {
                            "id": article_id,
                            "title": article_data["article"]["title"],
                            "language": lang,
                            "content": article_data["article"]["body"],
                        }
                    )

            except Exception as e:
                print(
                    f"Failed to get article {article_id} for language {lang}, error: {e}"
                )
                errors.append({"language": lang, "id": article_id})

    def load_sitemap(self):
        """
        Load language and article data from the sitemap.
        """
        root = ET.parse(self.sitemap_file).getroot()
        # Should probably find a way not to hardcode this, too
        prefix = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
        urls = []
        for child in root:
            urls.append(child.find(prefix + "loc").text)

        self.languages = set(map(self.get_lang_code, urls))
        self.languages.remove(None)

        self.article_ids = set(map(self.get_article_code, urls))
        self.article_ids.remove(None)

    def retry_failures(self):
        """
        Retried scraping on known failures
        """
        rate_limit = asyncio.Semaphore(2)  # limit requests per second
        rows = []
        # Since I hardcoded the failure file I know exactly where it is!
        sitemap_file = "failed_articles.csv"
        df_fails = pd.read_csv(sitemap_file, index_col=0)
        pairs = zip(df_fails["language"], df_fails["id"])
        double_errors = []

        async def run_tasks():
            async with aiohttp.ClientSession() as session:
                tasks = []

                for lang, article_id in pairs:
                    task = asyncio.ensure_future(
                        self.fetch_and_save(
                            session, rate_limit, lang, article_id, rows, double_errors
                        )
                    )
                    tasks.append(task)

                await asyncio.gather(*tasks)

            pd.DataFrame.from_records(rows).to_csv("retried_articles.csv", index=False)
            pd.DataFrame.from_records(double_errors).to_csv("double_failures.csv")

        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_tasks())

    def __call__(self, limit: int = 1):
        """
        Perform the actual scraping.
        """
        self.load_sitemap()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.gather_data(self.languages, self.article_ids, limit)
        )
