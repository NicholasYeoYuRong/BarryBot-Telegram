from dotenv import load_dotenv
from threading import Thread
from collections import defaultdict
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
import threading
from pathlib import Path
from datetime import datetime
from telebot import types
import requests
import time
import subprocess
import signal


# Load environment viariables
load_dotenv()

API_TOKEN = os.environ["TELEGRAM_TOKEN"]
MODEL_NAME = os.environ["MODEL_NAME"]
ICAL_URL = os.environ["ICAL_URL"]

BOT = telebot.TeleBot(token=API_TOKEN)
MODEL = MODEL_NAME

mcpmanager = None

class McpManager:
    def __init__(self):
        self.session = None
        self.loop = asyncio.new_event_loop()
        self.connected = threading.Event()
        self.tools: List[Any] = []
        self.server_process = None # Added server process
        self.server_thread = None

    async def _connect(self , server_params: StdioServerParameters):
        """Async connection handler"""
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()

                    # Cache tools list
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    print("\nConnected to server with tools:")
                    for tool in self.tools:
                        print(f" - {tool.name}: {tool.description}")

                    self.connected.set()

                    # Keep conenction alive
                    while True:
                        await asyncio.sleep(1)
        
        except Exception as e:
            print(f"MCP connection error: {str(e)}")
            self.connected.clear()
            if self.server_process:
                self.server_process.terminate()
                self.server_process = None

    def start_server(self):
        """Start the MCP server process"""
        if self.server_process is None or self.server_process.poll() is not None:
            self.server_process = subprocess.Popen(
                ["python", "mcp-server.py"],
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("Started MCP server process")
    
    def start_connection(self):
        """Start connection in background thread"""
        server_params = StdioServerParameters(
            command="python",
            args=["mcp-server.py"],
            cwd=str(Path.cwd())
        )

        def run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._connect(server_params))

        if not self.server_thread or not self.server_thread.is_alive():
            self.start_server() # Start the server if not already running
            self.server_thread = threading.Thread(target=run, daemon=True)
            self.server_thread.start()

    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]):
        """Synchronous wrapper for tool calls"""
        if not self.connected.is_set():
            raise ConnectionError("Not connected to MCP server")

        future = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(tool_name, arguments=arguments),
            self.loop
        )    
        return future.result()
    
    def shutdown(self):
        """Cleanup resources"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
        if self.loop:
            self.loop.stop()
    
mcp_manager = McpManager()
mcp_manager.start_connection()

# STORING CONVO HISTORY
conversation_history = defaultdict(list)


def extract_datetime(event_text):
    return event_text.split(" | ")[-1]

def format_event(event_text):
    title_part, duration_part, description_part, location_part, datetime_part, = event_text.split(" | ", 4)

    event_time = datetime.fromisoformat(datetime_part.strip())

    formatted_date = event_time.strftime("%d %B %Y")
    formatted_time = event_time.strftime("%H:%M")
    day_of_week = event_time.strftime("%A")

    return f"""Event: {title_part}
  Date: {formatted_date} ({day_of_week})
  Time: {formatted_time}
  Duration: {duration_part}
  Description: {description_part}
  Location: {location_part}"""

def format_inline_event(event_text):
    title_part, datetime_part = event_text.split(" | ")

    event_time = datetime.fromisoformat(datetime_part)

    formatted_date = event_time.strftime("%d %B %Y")
    formated_time = event_time.strftime("%H:%M")
    day_of_week = event_time.strftime("%A")

    return f"""{title_part}, {formatted_date}, {day_of_week} @ {formated_time}"""

def extract_event_name(event_text: str) -> str:
    return event_text.split(" | ")[0] ## CHANGE TO " | "

def extract_event_time(event_text: str) -> str:
    return event_text.split(" | ")[-1] ## CHANGE TO " | "

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
        if not mcp_manager.connected.is_set():
            BOT.send_message("⚠️ Connecting to server...")
            return
        
        items = "\n".join([f" - {tool.name}: {tool.description}"
                           for tool in mcp_manager.tools])
        BOT.send_message(message.chat.id, f"Available tools:\n{items}")

    except Exception as e:
        print(f"🚫 Unexpected error: {str(e)}")

@BOT.message_handler(commands=['addition'])
def addition_tool(message):
    try:
        
        prompt = message.text.replace('/addition', '').strip()
        number = prompt.split('+')
        if not prompt:
            BOT.send_message(message.chat.id, "Please specify the 2 number addition equation. Eg. '1 + 1' ")
            return
        
        a = int(number[0].strip())
        b = int(number[1].strip())

        result = mcp_manager.call_tool_sync("add", {"a": a, "b": b})
        BOT.send_message(message.chat.id, f"{a} + {b} = {result.content[0].text}")


    except Exception as e:
        print(f"🚫 Unexpected error: {str(e)}")

@BOT.message_handler(commands=['myevents'])
def list_calendar(message):
    try:
        
        result = mcp_manager.call_tool_sync("get_ical_events", {
            "ical_url": ICAL_URL,
            "max_results": 6
        })

        current_time  = datetime.now().astimezone()

        data = [item for item in result.content]

        future_events = []
        for event in data:
            event_time_str = extract_datetime(event.text)
            event_time = datetime.fromisoformat(event_time_str)
            if event_time > current_time:
                future_events.append((event_time, event.text))
        
        future_events.sort()

        formatted_events = []

        if not future_events:
            formatted_events.append(
                "Your schedule is free! There are no upcoming events!"
            )
        else:
            for i, (event_time, event_text) in enumerate(future_events, 1):
                formatted_events.append(
                    f"SCHEDULE {i}:\n"
                    f"  {format_event(event_text)}")

        structured_events = "\n\n".join(formatted_events)

        BOT.send_message(message.chat.id, f"===YOUR UPCOMING SCHEDULES===\n\n{structured_events}")
        

    except Exception as e:
        print(f"🚫 Unexpected error: {str(e)}")

# Add with other state tracking variables
user_states = {}  # Track conversation state
event_data = {}    # Store temporary event data

### DELETE EVENTS FROM CALENDAR ###
@BOT.message_handler(commands=['deleteEvent'])
def start_delete_event(message):
    chat_id = message.chat.id

    try:
        result = mcp_manager.call_tool_sync("get_all_ical_events", {
            "ical_url": ICAL_URL
        })

        current_time = datetime.now().astimezone()
        data = [item for item in result.content]

        future_events = []
        for event in data:
            event_time_str = extract_datetime(event.text)
            event_time = datetime.fromisoformat(event_time_str)
            if event_time > current_time:
                future_events.append((event_time, event.text))

        if not future_events:
            return BOT.send_message(chat_id, "No upcoming events found to delete")
        
        future_events.sort()

        markup = types.InlineKeyboardMarkup()
        for _, event_text in future_events:
            btn_text = format_inline_event(event_text)
            event_name = extract_event_name(event_text)
            event_time = extract_event_time(event_text)
            callback_data = f"delete_{event_name}_{event_time}"
            markup.add(types.InlineKeyboardButton(btn_text, callback_data=callback_data))
        
        markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_delete"))

        BOT.send_message(
            chat_id,
            "==============DELETE AN EVENT==============\n\n"
            "🗑️ Which event do you want to remove?",
            reply_markup=markup
        )

    except Exception as e:
        BOT.send_message(chat_id, f"❌ Error loading events: {str(e)}")

@BOT.callback_query_handler(func=lambda call: call.data == "cancel_delete")
def cancel_delete(call):
    chat_id = call.message.chat.id
    BOT.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="Deletion cancelled. You can start over with /deleteEvent"
    )

@BOT.callback_query_handler(func=lambda call: call.data.startswith('delete_'))
def handle_delete(call):
    chat_id = call.message.chat.id
    try:
        _, event_name, event_time = call.data.split('_', 2)

        result = mcp_manager.call_tool_sync("delete_calendar_event", {
            "event_name": event_name,
            "event_time": event_time
        })

        BOT.edit_message_text(
            chat_id=chat_id,
            message_id=call.message.message_id,
            text=f"{result.content[0].text.split(" @ ")[0]}\n\n"
                f"Event: {event_name}\n"
                f"Time: {datetime.fromisoformat(event_time).strftime('%Y-%m-%d  %H:%M')}"
        )

    except Exception as e:
        BOT.answer_callback_query(call.id, f"Error: {str(e)}", show_alert=True)

### ADDING EVENTS TO CALENDAR ###
@BOT.message_handler(commands=['addevent'])
def start_add_event(message):
    chat_id = message.chat.id
    user_states[chat_id] = 'awaiting_event_name'
    event_data[chat_id] = {'duration': 1.0}  # Set default duration

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event"))
    
    askmsg = BOT.send_message(
        chat_id,
        "Let's add an event!\n\n"
        "Please send me the event name:",
        reply_markup=markup
    )

@BOT.message_handler(func=lambda message: user_states.get(message.chat.id) == 'awaiting_event_name')
def handle_event_name(message):
    chat_id = message.chat.id
    event_data[chat_id]['name'] = message.text
    user_states[chat_id] = 'awaiting_datetime'

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event"))

    askmsg = BOT.send_message(
        chat_id,
        "📅 When is this happening?\n"
        "Examples:\n"
        "- Tomorrow 2pm OR 2:30pm \n"
        "- 2023-12-25 14:30\n"
        "- Next Friday 3pm OR 3:30pm\n"
        "- Sunday 5pm OR 5.30pm",
        reply_markup=markup
    )

@BOT.message_handler(func=lambda message: user_states.get(message.chat.id) == 'awaiting_datetime')
def handle_datetime(message):
    chat_id = message.chat.id
    event_data[chat_id]['start_time'] = message.text
    user_states[chat_id] = 'awaiting_duration'
    
    # Create inline skip button
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Skip (use 1 hour)", callback_data="skip_duration"))
    markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event"))
    
    BOT.send_message(
        chat_id,
        "⏳ How long will it last in hours? (Default: 1 hour)",
        reply_markup=markup
    )

@BOT.callback_query_handler(func=lambda call: call.data == "skip_duration")
def skip_duration(call):
    chat_id = call.message.chat.id
    user_states[chat_id] = 'awaiting_description'
    
    # Create inline skip button
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Skip description", callback_data="skip_description"))
    markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event"))
    
    BOT.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="⏳ Duration set to default 1 hour"
    )
    
    BOT.send_message(
        chat_id,
        "📝 Any description? (Optional)",
        reply_markup=markup
    )

@BOT.message_handler(func=lambda message: user_states.get(message.chat.id) == 'awaiting_duration')
def handle_duration(message):
    chat_id = message.chat.id
    try:
        event_data[chat_id]['duration'] = float(message.text)
    except ValueError:
        BOT.send_message(chat_id, "⚠️ Please send a number (like 1 or 1.5)")
        return
    
    user_states[chat_id] = 'awaiting_description'
    
    # Create inline skip button
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Skip description", callback_data="skip_description"))
    markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event"))
    
    BOT.send_message(
        chat_id,
        "📝 Any description? (Optional)",
        reply_markup=markup
    )

@BOT.callback_query_handler(func=lambda call: call.data == "skip_description")
def skip_description(call):
    chat_id = call.message.chat.id
    event_data[chat_id]['description'] = ""
    user_states[chat_id] = 'awaiting_location'
    
    # Create inline skip button
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Skip location", callback_data="skip_location"))
    markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event"))
    
    BOT.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="📝 Description skipped"
    )
    
    BOT.send_message(
        chat_id,
        "📍 Location? (Optional)",
        reply_markup=markup
    )

@BOT.message_handler(func=lambda message: user_states.get(message.chat.id) == 'awaiting_description')
def handle_description(message):
    chat_id = message.chat.id
    event_data[chat_id]['description'] = message.text
    user_states[chat_id] = 'awaiting_location'
    
    # Create inline skip button
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Skip location", callback_data="skip_location"))
    markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event"))
    
    BOT.send_message(
        chat_id,
        "📍 Location? (Optional)",
        reply_markup=markup
    )

@BOT.callback_query_handler(func=lambda call: call.data == "skip_location")
def skip_location(call):
    chat_id = call.message.chat.id
    event_data[chat_id]['location'] = ""
    
    BOT.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="📍 Location skipped"
    )
    
    confirm_and_add_event(chat_id)

@BOT.message_handler(func=lambda message: user_states.get(message.chat.id) == 'awaiting_location')
def handle_location(message):
    chat_id = message.chat.id
    event_data[chat_id]['location'] = message.text
    confirm_and_add_event(chat_id)

def confirm_and_add_event(chat_id):
    # Format confirmation message
    event = event_data[chat_id]
    confirm_msg = (
        "✅ Please confirm:\n\n"
        f"Event: {event['name']}\n"
        f"Time: {event['start_time']}\n"
        f"Duration: {event['duration']} hours\n"
        f"Description: {event.get('description', 'None')}\n"
        f"Location: {event.get('location', 'None')}\n\n"
        "Is this correct?"
    )
    
    # Send confirmation with buttons
    markup = types.InlineKeyboardMarkup()
    markup.add(
        types.InlineKeyboardButton("Yes", callback_data="event_confirm_yes"),
        types.InlineKeyboardButton("No", callback_data="event_confirm_no")
    )
    
    BOT.send_message(chat_id, confirm_msg, reply_markup=markup)
    user_states[chat_id] = 'awaiting_confirmation'

@BOT.callback_query_handler(func=lambda call: call.data.startswith('event_confirm_'))
def handle_confirmation(call):
    chat_id = call.message.chat.id
    if call.data == "event_confirm_yes":
        try:
            # Call your MCP tool
            result = mcp_manager.call_tool_sync("add_ical_event", {
                "event_name": event_data[chat_id]['name'],
                "start_time": event_data[chat_id]['start_time'],
                "duration_hours": event_data[chat_id]['duration'],
                "description": event_data[chat_id].get('description', ''),
                "location": event_data[chat_id].get('location', '')
            })
            
            BOT.send_message(chat_id, result.content[0].text)
        except Exception as e:
            BOT.send_message(chat_id, f"❌ Error adding event: {str(e)}")
    else:
        BOT.send_message(chat_id, "Event cancelled. Start over with /addevent")
    
    # Clean up
    user_states.pop(chat_id, None)
    event_data.pop(chat_id, None)
    BOT.delete_message(chat_id, call.message.message_id)

@BOT.callback_query_handler(func=lambda call: call.data == "cancel_add_event")
def cancel_add_event(call):
    chat_id = call.message.chat.id
    user_states.pop(chat_id, None)
    event_data.pop(chat_id, None)
    
    BOT.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="Event creation cancelled. You can start over with /addevent"
    )



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

        # DIFFUSER_MODEL = "sd-legacy/stable-diffusion-v1-5" ## TEMPORARY MODEL FOR NOW, TO USE black-forest-labs/FLUX.1-dev IN FUTURE ##

        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # torch.backends.cuda.enable_flash_sdp(True) 

        # # Load pipeline with optimizations
        # pipe = StableDiffusionPipeline.from_pretrained(DIFFUSER_MODEL, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
        # pipe = pipe.to("cuda")
        # torch.cuda.empty_cache()
        
        # image = pipe(
        #     prompt=prompt,
        #     height=512,  
        #     width=512,
        # ).images[0]

        # Call Stability AI API
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={
                "Authorization": f"Bearer {os.getenv('STABILITY_API_KEY')}",
                "Accept": "image/*"
            },
            files={"none": ''},
            data={
                "prompt": prompt,
                "output_format": "png"
            },
            timeout=30  # 30-second timeout
        )

        # Convert to Bytes
        img_bytes = BytesIO(response.content)

        # Convert to bytes and send
        # img_bytes = BytesIO()
        # response.save(img_bytes, format='PNG')
        # img_bytes.seek(0)
        
        # Send the image
        BOT.delete_message(chat_id=message.chat.id, message_id=wait_msg.message_id)
        BOT.send_photo(message.chat.id, img_bytes)
        # BOT.send_photo(message.chat.id, img_bytes)
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



def run_bot():
    while True:
        try:
            print("Starting bot...")
            BOT.polling(non_stop=True, skip_pending=True)
        except Exception as e:
            print(f"Bot crashed: {str(e)}")
            time.sleep(10)  # Wait before restarting

def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    mcp_manager.shutdown()
    os._exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start bot in main thread
    run_bot()
