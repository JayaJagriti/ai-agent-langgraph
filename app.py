from agent import create_agent

agent = create_agent()

print("🤖 AI Agent is live! Ask me anything (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    response = agent.invoke({"query": query})
    print("AI:", response["result"])