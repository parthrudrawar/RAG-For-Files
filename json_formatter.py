import json
from typing import Dict, List
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()

class JSONFormatter:
    def __init__(self, input_file: str = "raw_data.json", output_file: str = os.getenv("FORMATTED_DATA_FILE", "formatted_data.json")):
        self.input_file = input_file
        self.output_file = output_file

    def load_raw_data(self) -> Dict:
        """Load raw JSON data."""
        try:
            with open(self.input_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'data': []}

    def fetch_link_title(self, url: str) -> str:
        """Fetch the title of a linked page."""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('h1') or soup.find('title')
            return title.get_text().strip() if title else url.split('/')[-1]
        except Exception:
            return url.split('/')[-1]

    def format_data(self) -> Dict:
        """Format raw data for embedding."""
        raw_data = self.load_raw_data()
        formatted_data = {'data': []}
        
        for item in raw_data['data']:
            formatted_item = {
                'url': item['url'],
                'title': item['title'],
                'related_sections': []
            }
            
            for link in item['links']:
                title = self.fetch_link_title(link)
                formatted_item['related_sections'].append({
                    'title': title,
                    'url': link,
                    'text': f"{title} ({link})"
                })
            
            formatted_data['data'].append(formatted_item)
        
        return formatted_data

    def save_formatted_data(self):
        """Save formatted data to JSON file."""
        formatted_data = self.format_data()
        with open(self.output_file, 'w') as f:
            json.dump(formatted_data, f, indent=2)

if __name__ == "__main__":
    formatter = JSONFormatter()
    formatter.save_formatted_data()