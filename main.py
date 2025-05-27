from config import API_TOKEN, MODEL_NAME
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from time import sleep
from threading import Thread
from collections import defaultdict
import ollama
import telebot

# Load environment viariables
load_dotenv()

BOT = telebot.TeleBot(token=API_TOKEN)
MODEL = MODEL_NAME

class TypingIndicator:
    def __init__(self, bot, chat_id):
        self.bot = bot
        self.chat_id = chat_id
        self._stop_event = False
        
    def run(self):
        while not self._stop_event:
            try:
                self.bot.send_chat_action(self.chat_id, 'typing')
                sleep(2.5)  # Send every 2.5 seconds (Telegram caches for 5s)
            except:
                break
                
    def stop(self):
        self._stop_event = True

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



BOT.polling()