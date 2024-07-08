import os
os.sys.path.append('..')
from src.engine import Engine
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI

api_key = os.getenv('API_KEY', 'EMPTY')
base_url = os.getenv('BASE_URL', 'http://localhost:8000/v1')
# engine = Engine('facebook/opt-125m', api_key, base_url)
engine = Engine('meta-llama/Meta-Llama-3-70B-Instruct', api_key, base_url)
api_key, base_url

print(engine.run("Hello", "hi"))