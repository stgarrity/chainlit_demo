import chainlit as cl
import openai
import os

api_key = os.getenv("OPENAI_API_KEY")

endpoint_url = "https://api.openai.com/v1"
client = openai.AsyncClient(api_key=api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {
    "model": "chatgpt-4o-latest",
    "temperature": 0.3,
    "max_tokens": 500
}

@cl.on_message
async def on_message(message: cl.Message):
    # Your custom logic goes here...

    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": message.content}],
        **model_kwargs
    )

    # https://platform.openai.com/docs/guides/chat-completions/response-format
    response_content = response.choices[0].message.content

    await cl.Message(
        content=response_content,
    ).send()
