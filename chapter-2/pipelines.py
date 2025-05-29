import torch
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset

import soundfile as sf

from IPython.display import Audio

classifier = pipeline("sentiment-analysis")

result = classifier("I've been waiting for a Hugging Face course my whole life.")

print(result)






