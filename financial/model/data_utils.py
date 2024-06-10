import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def download_raw_texts_from_urls(current_urls):
    raw_texts = []
    for url in tqdm(current_urls, total=len(current_urls)):
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        parsed_text = ' '.join(soup.text.replace('\n', " ").strip().split())
        raw_texts.append(parsed_text)
    return raw_texts

def save_text_to(data, file_path):
    with open(file_path, 'w') as file:
        if type(data) == str:
            file.write(data)
        elif type(data) == list:
            for line in data:
                file.write(line + "\n")
                file.write("<SEP>" + "\n")

def load_text_from(file_path):
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    return data