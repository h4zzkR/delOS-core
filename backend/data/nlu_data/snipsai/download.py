from urllib.request import urlretrieve
from pathlib import Path
import os


SNIPS_DATA_BASE_URL = (
    "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
    "master/data/snips/"
)

for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:
    path = Path(os.path.join('./data', filename))
    if not path.exists():
        print(f"Downloading {filename}...")
        urlretrieve(SNIPS_DATA_BASE_URL + filename + "?raw=true", path)