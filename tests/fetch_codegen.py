# -*- coding: utf-8 -*-
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

print("OK", "codegen-350M-mono", "->",
      snapshot_download("Salesforce/codegen-350M-mono"))
