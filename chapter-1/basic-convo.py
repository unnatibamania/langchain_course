from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)



chat_history = []


system_message = SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks.")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    result = model.invoke(chat_history)
    chat_history.append(result)
    print(f"AI: {result.content}")
    

print("Goodbye!")