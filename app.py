from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.document_loaders import GithubFileLoader

loader = GithubFileLoader(
    repo="langchain-ai/langchain",  
    branch="master",  
    github_api_url="https://api.github.com",
    file_filter=lambda file_path: file_path.endswith(".md"),  
)
docs = loader.load()
print(len(docs))