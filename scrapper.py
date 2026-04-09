import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
from dotenv import load_dotenv
import os
from urllib.parse import urljoin

load_dotenv()

class WebScraper:
    def __init__(self, urls_file: str = os.getenv("URLS_FILE", "url.txt")):
        self.urls_file = urls_file
        self.data = []

    def load_urls(self) -> List[str]:
        """Load URLs from file, filter invalid entries, or return sample URLs."""
        try:
            with open(self.urls_file, 'r') as file:
                urls = [line.strip() for line in file if line.strip() and line.strip().startswith('http')]
                if not urls:
                    print("Warning: url.txt is empty or contains no valid URLs. Using sample URLs.")
                    return ["https://www.wikipedia.org", "https://www.example.com", "https://www.python.org"]
                return urls
        except FileNotFoundError:
            print("Error: url.txt not found. Using sample URLs.")
            return ["https://www.wikipedia.org", "https://www.example.com", "https://www.python.org"]

    def scrape_page(self, url: str) -> Dict:
        """Scrape a single page for links and title."""
        print(f"Scraping URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title
            title = soup.find('h1') or soup.find('title')
            title_text = title.get_text().strip() if title else url.split('/')[-1]

            # Extract related links
            links = []
            base_url = url.rsplit('/', 1)[0]
            for a in soup.find_all('a', href=True):
                href = a['href']
                if not href.startswith('http'):
                    href = urljoin(base_url, href)
                if href.startswith(('http://', 'https://')):
                    links.append(href)

            return {
                'url': url,
                'title': title_text,
                'links': list(set(links))
            }
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {'url': url, 'title': url.split('/')[-1], 'links': []}

    def scrape_all(self) -> List[Dict]:
        """Scrape all URLs and store results."""
        self.data = [self.scrape_page(url) for url in self.load_urls()]
        return self.data

    def save_raw_data(self, output_file: str = "raw_data.json"):
        """Save scraped data to JSON file."""
        print(f"Saving data to {output_file}")
        with open(output_file, 'w') as f:
            json.dump({'data': self.data}, f, indent=2)

if __name__ == "__main__":
    scraper = WebScraper()
    scraper.scrape_all()
    scraper.save_raw_data()