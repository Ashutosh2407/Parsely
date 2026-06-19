import weaviate
import os
import json
from dotenv import load_dotenv

load_dotenv()

headers = {
    "X-Cohere-Api-Key": os.environ.get("COHERE_API_KEY")
}

try:
    client = weaviate.connect_to_local(headers=headers)
    assert client.is_ready()
    metainfo = client.get_meta()
    print(json.dumps(metainfo, indent=2))
    print("Health: OK")
finally:
    client.close()