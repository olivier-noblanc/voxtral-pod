import requests
import sys

url = "https://archive.org/download/fables_de_la_fontaine_01_1009_librivox/fables_delafontaine_01_01_lafontaine_64kb.mp3"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

try:
    print(f"Downloading {url}...")
    r = requests.get(url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()
    with open("tests/samples/french_sample.mp3", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
