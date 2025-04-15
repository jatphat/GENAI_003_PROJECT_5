import requests
from bs4 import BeautifulSoup

def scrape_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove all script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Get the text from the HTML
        text = soup.get_text()
        # Break the text into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        text = None
        print(f"An error occurred: {e}")
    return text