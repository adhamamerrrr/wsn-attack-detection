"""Download the authors' public Kaggle WSN-DS archive; commit no dataset."""
from pathlib import Path
from urllib.request import urlopen
import zipfile
import io

URL = 'https://www.kaggle.com/api/v1/datasets/download/bassamkasasbeh1/wsnds'

def main():
    destination = Path('data')
    destination.mkdir(exist_ok=True)
    try:
        with urlopen(URL, timeout=90) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        candidates = [n for n in archive.namelist() if n.lower().endswith('.csv')]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one CSV')
        # Use a fixed filename instead of trusting archive paths.
        (destination/'WSN-DS.csv').write_bytes(archive.read(candidates[0]))
        print('Saved data/WSN-DS.csv; see DATASET.md for attribution and terms.')
    except Exception as exc:
        raise SystemExit(f'Download failed: {exc}. Download manually from https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds and save data/WSN-DS.csv. Never put credentials in this repository.')

if __name__ == '__main__':
    main()
