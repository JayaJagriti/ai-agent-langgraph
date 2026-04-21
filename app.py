from agent import create_agent

agent = create_agent()

print("🤖 Autonomous AI Agent (type 'exit')")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    response = agent.invoke({
        "query": query,
        "history": [],
        "scratchpad": "",
        "step": 0
    })

    print("AI:", response["result"])