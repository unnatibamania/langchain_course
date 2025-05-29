import dotenv
import os

from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

result = model.invoke("what is the capital of the moon?")

print(result)