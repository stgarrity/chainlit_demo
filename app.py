import base64
import copy
import chainlit as cl
import openai
import os

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth

api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = "https://api.openai.com/v1"
# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
    "model": "chatgpt-4o-latest",
    "temperature": 0.2,
    "max_tokens": 500
}

wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                    
    auth_credentials=Auth.api_key(wcd_api_key),
)

# FIXME - why didn't my index load from jupyter work?
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core.node_parser import SimpleNodeParser
# docs = SimpleDirectoryReader('./data').load_data()
# parser = SimpleNodeParser()
# vector_store = WeaviateVectorStore(weaviate_client = weaviate_client, index_name="Tesla")
# nodes = parser.get_nodes_from_documents(docs)
# storage_context = StorageContext.from_defaults(vector_store = vector_store)
# index = VectorStoreIndex(nodes, storage_context = storage_context)
# query_engine = index.as_query_engine()
# response = query_engine.query("reasons that my tesla's velocity is lower than usual")
# print(response)

vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name="Tesla")
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine()
retriever = index.as_retriever()

# api_key = os.getenv("RUNPOD_API_KEY")
# endpoint_url = f"https://api.runpod.ai/v2/{runpod_serverless_id}/openai/v1"
# runpod_serverless_id = os.getenv("RUNPOD_SERVERLESS_ID")
# model_kwargs = {
#     "model": "mistralai/Mistral-7B-Instruct-v0.3",
#     "temperature": 0.3,
#     "max_tokens": 500
# }

client = wrap_openai(openai.AsyncClient(api_key=api_key, base_url=endpoint_url))

@traceable
@cl.on_message
async def on_message(message: cl.Message):
    # Maintain an array of messages in the user session
    message_history = cl.user_session.get("message_history", [])
    
    # Processing images exclusively
    images = [file for file in message.elements if "image" in file.mime] if message.elements else []

    if images:
        # Read the first image and encode it to base64
        with open(images[0].path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        message_history.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message.content if message.content else "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    else:
        message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()

    rag_history = copy.deepcopy(message_history)
    rag_history.append({"role": "system", "content": "Your only job is to identify if you need extra information from the Tesla Cyber Truck's Owners Manual to answer the last message in this thread. Respond with only one word, yes or no."})
    
    rag = await client.chat.completions.create(messages=rag_history, **model_kwargs)
    if rag.choices[0].message.content.lower() == "yes":
        print("retrieving data")
        chunks = retriever.retrieve(message.content)
        
        context = ""
        for chunk in chunks:
            context += chunk.text

        message_history[len(message_history)-1]["content"] += context

    # Pass in the full message history for each request
    stream = await client.chat.completions.create(messages=message_history, 
                                                stream=True, **model_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)