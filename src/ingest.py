import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

DIR_PATH = "data/"

def load_all_docs(data_dir: str)-> list:
    all_docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir,filename))
            docs = loader.load()
            print(f"{filename}: {len(docs)} pages.")
            all_docs.extend(docs)
    print(f"\nTotal doc chunks loaded: {len(all_docs)}")
    return all_docs

if __name__ == "__main__":
    load_all_docs(DIR_PATH)

