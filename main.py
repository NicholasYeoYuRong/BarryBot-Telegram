from config import API_TOKEN, MODEL_NAME
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from threading import Thread
from collections import defaultdict
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler, FluxPipeline
import torch
import requests
from io import BytesIO
import ollama
import telebot
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import asyncio
from typing import Any, Dict, List
from tele_indicators import TypingIndicator, SendingPhotoIndicator


# Load environment viariables
load_dotenv()

BOT = telebot.TeleBot(token=API_TOKEN)
MODEL = MODEL_NAME


session = None
stdio = None
write = None
exit_stack = AsyncExitStack()

# Connection to server #
async def connect_to_server(server_script_path: str = "mcp-server.py"):
    """Connect to an MCP server.
    
    Args:
        server_scrip_path: Path to the server script.
    """
    global session, stdio, write, exit_stack

    #Server configuration
    server_params = StdioServerParameters(
        command="python",
        args=[server_script_path],
    )

    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    # Initialize the connection
    await session.initialize()

    # List available tools
    tools_result = await session.list_tools()
    print("\nConnected to server with tools:")
    for tool in tools_result.tools:
        print(f" - {tool.name}: {tool.description}")

# Handling task #
async def handle_list_items(message):
    try:
        await connect_to_server("mcp-server.py")

        tools_result = await session.list_tools()
        items = "\n".join([f" - {tool.name}: {tool.description}" for tool in tools_result.tools])

        BOT.send_message(message.chat.id, f"Available tools:\n{items}")

        await cleanup()
    except Exception as e:
        print(f"🚫 Unexpected error in async handler: {str(e)}")



# STORING CONVO HISTORY
conversation_history = defaultdict(list)

# /start #
@BOT.message_handler(commands=['start'])
def welcome(message):
    welcome_text = f'Hi {message.from_user.first_name}, My name is Barry! How can I assist you today?'
    BOT.send_message(message.chat.id, welcome_text)

@BOT.message_handler(commands=['reset'])
def reset_chat(message):
    conversation_history[message.chat.id] = []
    BOT.send_message(message.chat.id, "***CHAT RESETTED***")

@BOT.message_handler(commands=['listItems'])
def list_Items(message):
    try:

        asyncio.run(handle_list_items(message))

    except Exception as e:
        print(f"🚫 Unexpected error: {str(e)}")


## GENERATING AN IMAGE NEED ANOTHER IMAGE GENERATION MODEL ##
## USE HUGGING FACE DIFFUSERS ###
@BOT.message_handler(commands=['image', 'draw'])
def generate_image(message):
    try:
        prompt = message.text.replace('/image', '').replace('/draw', '').strip()
        if not prompt:
            BOT.reply_to(message, "Please describe what you want me to draw after the command")
            return
        
        # Show typing indicator
        sending = SendingPhotoIndicator(BOT, message.chat.id)
        s = Thread(target=sending.run)
        s.start()

        wait_msg = BOT.reply_to(message, "🖌️ Generating your image... (30-60 seconds)")

        DIFFUSER_MODEL = "sd-legacy/stable-diffusion-v1-5" ## TEMPORARY MODEL FOR NOW, TO USE black-forest-labs/FLUX.1-dev IN FUTURE ##

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.backends.cuda.enable_flash_sdp(True) 

        # Load pipeline with optimizations
        pipe = StableDiffusionPipeline.from_pretrained(DIFFUSER_MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
        pipe = pipe.to("cuda")
        torch.cuda.empty_cache()
        
        image = pipe(
            prompt=prompt,
            height=512,  
            width=512,
        ).images[0]

        # Convert to bytes and send
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        BOT.delete_message(chat_id=message.chat.id, message_id=wait_msg.message_id)
        BOT.send_photo(message.chat.id, img_bytes)
        sending.stop()
        s.join()
    
    except Exception as e:
        BOT.reply_to(message, f"🚫 Unexpected error: {str(e)}")


# REPLYING TO USER MESSAGE #
@BOT.message_handler(func=lambda message:True)
def reply_func(message):
    try:

        # START TYPING INDICATOR #
        typing = TypingIndicator(BOT, message.chat.id)
        t = Thread(target=typing.run)
        t.start()

        # Get or initialize conversation history for this chat
        chat_id = message.chat.id
        if chat_id not in conversation_history:
            conversation_history[chat_id][-6:] = [
                {'role': 'system', 'content': "Keep response concise"}
            ]

        # Add user message to history
        conversation_history[chat_id].append(
            {'role': 'user', 'content': message.text}
        )

        response = ollama.chat(
            model=MODEL,
            messages=conversation_history[chat_id]
        )

        # Add assistant response to history
        response_text = response['message']['content']
        conversation_history[chat_id].append(
            {'role': 'assistant', 'content': response_text}
        )

        for i in range(0, len(response_text), 4000):
            chunk = response_text[i:i+4000]
            if i == 0:
                BOT.reply_to(message, chunk)
            else:
                BOT.send_message(message.chat.id, chunk)
    

    except Exception as e:
        print(f"Error processing message: {e}")
        BOT.reply_to(message, "Sorry, I encountered an error processing your request.")

    finally:
        if typing:
            typing.stop()
            t.join()

async def cleanup():
    """Clean up resources."""
    global exit_stack
    await exit_stack.aclose()

if __name__ == "__main__":
    BOT.polling()
