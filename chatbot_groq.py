from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# -----------------------------
# Initialize Model
# -----------------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key
)

# -----------------------------
# Create Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. Answer clearly and politely."
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        "{question}"
    )
])

# -----------------------------
# Initialize Chat History
# -----------------------------
history = ChatMessageHistory()

print("=" * 60)
print("🤖 AI Chatbot")
print("Type 'exit' to quit")
print("=" * 60)

# -----------------------------
# Chat Loop
# -----------------------------
while True:

    user_input = input("\nYou : ")

    if user_input.lower() == "exit":
        print("\nGoodbye 👋")
        break

    # Create Prompt
    final_prompt = prompt.invoke({
        "chat_history": history.messages,
        "question": user_input
    })

    # Get Response
    response = model.invoke(final_prompt)

    # Display Response
    print("\nAI :", response.content)

    # Store Conversation
    history.add_user_message(user_input)
    history.add_ai_message(response.content)