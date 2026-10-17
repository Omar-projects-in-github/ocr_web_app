from huggingface_hub import hf_hub_download
import json

p = hf_hub_download("datalab-to/chandra-ocr-2", "config.json")
cfg = json.load(open(p))
print(json.dumps(cfg, indent = 2)[:4000])