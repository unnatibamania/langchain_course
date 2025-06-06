import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
from items import Item
import matplotlib.pyplot as plt

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
hf_token = os.getenv("HF_TOKEN")

print(hf_token)
login(token=hf_token, add_to_git_credential=True)


