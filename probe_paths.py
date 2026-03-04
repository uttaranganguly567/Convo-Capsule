from huggingface_hub import hf_hub_url
import requests

repo_id = "ai4bharat/Kathbath"
languages = ["hindi"]
patterns = [
    "{lang}/train.tar.gz",
    "data/{lang}/train.tar.gz",
    "{lang}.tar.gz",
    "data/{lang}.tar.gz",
    "audio/{lang}/train.tar.gz",
    "{lang}/audio.tar.gz",
    "release/{lang}/train.tar.gz"
]

print(f"Probing paths for {repo_id}...")

for lang in languages:
    for pat in patterns:
        filename = pat.format(lang=lang)
        url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
        
        # HEAD request to check existence (fast)
        try:
            r = requests.head(url, allow_redirects=True)
            if r.status_code == 200 or r.status_code == 302:
                print(f"✅ FOUND! {filename}")
                print(f"URL: {url}")
                exit(0)
            else:
                print(f"❌ {filename} (Status: {r.status_code})")
        except Exception as e:
            print(f"Error checking {filename}: {e}")

print("Could not find file in likely paths.")
