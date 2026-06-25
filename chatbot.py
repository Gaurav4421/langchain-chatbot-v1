from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.4
    }
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="""
You are an AI assistant.
You are to answer the following question in a {style} style.
You do not have personal experiences.
Answer truthfully.

Question: {question}
""",
    input_variables=["question", "style"]
)
chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break
    
    chat_history.append(user_input)
    result = model.invoke(chat_history)
    
    prompt = template.invoke({
        "question": user_input,
        "style": "complex"
    })

    result = model.invoke(prompt)

    print("AI:", result.content)