# -*- coding: utf-8 -*-
import os
import traceback

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

for repo in ["jingyaogong/minimind-3", "Salesforce/codegen-25M-mono"]:
    try:
        print("OK", repo, "->", snapshot_download(repo))
    except Exception as e:
        print("FAIL", repo, ":", repr(e)[:300])
        traceback.print_exc()
