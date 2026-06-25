from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature":0.2,
        "max_new_tokens":200
    }
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="""
You are a {role}.

Answer the following question in {style} style.

Question: the question is who was{name}
""",
    input_variables=["role","style","name"]
)

prompt = template.invoke({
    "role":"History Teacher",
    "style":"complex",
    "name":"Ashoka?"
})
result = model.invoke(prompt)
print(result.content)