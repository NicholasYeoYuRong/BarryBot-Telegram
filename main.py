from dotenv import load_dotenv
from threading import Thread
from collections import defaultdict
from io import BytesIO
import telebot
import os
from tele_indicators import TypingIndicator, SendingPhotoIndicator
from datetime import datetime
from telebot import types
import requests
import time
from openai import OpenAI
from telegram_bot_calendar import DetailedTelegramCalendar, LSTEP
from time_picker import TimePicker

from ical_handler import (
    get_ical_events,
    get_all_ical_events,
    add_ical_event,
    delete_calendar_event,
)


# Load environment viariables
load_dotenv()

API_TOKEN = os.environ["TELEGRAM_TOKEN"]
ICAL_URL = os.environ["ICAL_URL"]
agent_endpoint = os.environ["agent_endpoint"] + "/api/v1/"
agent_access_key = os.environ["agent_access_key"]

ALLOWED_USERS = ["Nicholas_yowo", "chzcookie"]

BOT = telebot.TeleBot(token=API_TOKEN)
time_picker = TimePicker()


# STORING CONVO HISTORY
conversation_history = defaultdict(list)

# Check if user is authorized
def is_allowed_user(message: types.Message) -> bool:
    return message.from_user.username in ALLOWED_USERS

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
    formatted_time = event_time.strftime("%H:%M")
    day_of_week = event_time.strftime("%A")

    return f"""{title_part}, {formatted_date}, {day_of_week} @ {formatted_time}"""

def extract_event_name(event_text: str) -> str:
    return event_text.split(" | ")[0] ## CHANGE TO " | "

def extract_event_time(event_text: str) -> str:
    return event_text.split(" | ")[-1] ## CHANGE TO " | "

# /start #
@BOT.message_handler(commands=['start'])
def welcome(message):
    if message.from_user.username == "Nicholas_yowo":
        BOT.send_message(message.chat.id, f"Hello, Creator {message.from_user.username}! How can I assist you today?")
    elif message.from_user.username == "chzcookie":
        BOT.send_message(message.chat.id, f"Hello, My Creator's Lovely Girlfriend, {message.from_user.first_name}! What do I owe this honour today?")
    else:
        welcome_text = f'Hi {message.from_user.first_name}, My name is Barry! How can I assist you today?'
        BOT.send_message(message.chat.id, welcome_text)

@BOT.message_handler(commands=['reset'])
def reset_chat(message):
    conversation_history[message.chat.id] = []
    BOT.send_message(message.chat.id, "***CHAT RESETTED***")

@BOT.message_handler(commands=['myevents'])
def list_calendar(message):
    try:

        events = get_ical_events(ICAL_URL, max_results=6)
        current_time = datetime.now().astimezone()

        future_events = []
        for event_text in events:
            event_time_str = extract_datetime(event_text)
            event_time = datetime.fromisoformat(event_time_str)
            if event_time > current_time:
                future_events.append((event_time, event_text))

        future_events.sort()

        formatted_events = []
        if not future_events:
            formatted_events.append("Your schedule is free! There are no upcoming events!")
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
@BOT.message_handler(commands=['deleteEvent'], func=is_allowed_user)
def start_delete_event(message):
    chat_id = message.chat.id

    try:
        result = get_all_ical_events(ICAL_URL)

        current_time = datetime.now().astimezone()

        future_events = []
        for event_text in result:
            event_time_str = extract_datetime(event_text)
            event_time = datetime.fromisoformat(event_time_str)
            if event_time > current_time:
                future_events.append((event_time, event_text))

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

        result = delete_calendar_event(event_name, event_time)

        BOT.edit_message_text(
            chat_id=chat_id,
            message_id=call.message.message_id,
            text=f"{result.split(' @ ')[0]}\n\n"
                f"Event: {event_name}\n"
                f"Time: {datetime.fromisoformat(event_time).strftime('%Y-%m-%d %H:%M')}"
        )

    except Exception as e:
        BOT.answer_callback_query(call.id, f"Error: {str(e)}", show_alert=True)

### ADDING EVENTS TO CALENDAR ###
@BOT.message_handler(commands=['addevent'], func=is_allowed_user)
def start_add_event(message):
    chat_id = message.chat.id
    user_states[chat_id] = 'awaiting_event_name'
    event_data[chat_id] = {'duration': 1.0}  # Set default duration

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event"))
    
    BOT.send_message(
        chat_id,
        "Let's add an event!\n\n"
        "Please send me the event name:",
        reply_markup=markup
    )

@BOT.message_handler(func=lambda message: user_states.get(message.chat.id) == 'awaiting_event_name')
def handle_event_name(message):
    chat_id = message.chat.id
    message_id = message.message_id
    event_data[chat_id]['name'] = message.text
    user_states[chat_id] = 'awaiting_datetime'

    calendar, step = DetailedTelegramCalendar().build()

    # markup = types.InlineKeyboardMarkup()
    # markup.add(types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event"))

    BOT.send_message(
        chat_id,
        "📅 When is this happening?\n\n"
        f"Select {LSTEP[step]}",
        reply_markup=calendar
    )

@BOT.callback_query_handler(func=DetailedTelegramCalendar.func())
def handle_calendar_query(call):
    chat_id = call.message.chat.id
    result, key, step = DetailedTelegramCalendar().process(call.data)

    # Check if the result is a valid date
    if not result and key:
        BOT.edit_message_text("📅 When is this happening?\n\n"
                            f"Select {LSTEP[step]}",
                              chat_id,
                              call.message.message_id,
                              reply_markup=key)
    elif result:
        markup = types.InlineKeyboardMarkup()
        markup.add(
            types.InlineKeyboardButton("Confirm", callback_data=f"confirm_date_{result}"),
            types.InlineKeyboardButton("Change Date", callback_data="change_date"),
        )

        BOT.edit_message_text(f"📅 Selected date: {result}",
                              chat_id,
                              call.message.message_id,
                              reply_markup=markup)

# Catches the "Change Date" callback
@BOT.callback_query_handler(func=lambda call: call.data == "change_date")
def change_date(call):
    chat_id = call.message.chat.id
    user_states[chat_id] = 'awaiting_datetime'

    calendar, step = DetailedTelegramCalendar().build()

    BOT.edit_message_text(
        "📅 When is this happening?\n\n"
        f"Select {LSTEP[step]}",
        chat_id,
        call.message.message_id,
        reply_markup=calendar
    )

@BOT.callback_query_handler(func=lambda call: call.data.startswith('confirm_date_'))
def confirm_date(call):
    chat_id = call.message.chat.id
    date_str = call.data.split('_', 2)[2]
    event_data[chat_id]['date_only'] = datetime.strptime(date_str, "%Y-%m-%d").date()

    user_states[chat_id] = 'awaiting_time'

    BOT.send_message(
        chat_id,
        "🕒 Select a time:",
        reply_markup=time_picker.create_time_picker(chat_id, event_data[chat_id]['date_only'])
    )

@BOT.callback_query_handler(func=lambda call: call.data.startswith('time_'))
def handle_time_selection(call):
    """Handle time picker interactions"""
    chat_id = call.message.chat.id
    message_id = call.message.message_id

    base_date = event_data[chat_id]['date_only']

    should_continue, selected_dt = time_picker.handle_callback(call, base_date)

    if should_continue:
        BOT.edit_message_reply_markup(
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=time_picker.create_time_picker(chat_id, base_date),
        )
    else:
        if selected_dt:
            event_data[chat_id]['datetime'] = selected_dt.strftime("%Y-%m-%d %H:%M")
            user_states[chat_id] = 'awaiting_duration'

            markup = types.InlineKeyboardMarkup()
            markup.add(
                types.InlineKeyboardButton("Skip (use 1 hour)", callback_data="skip_duration"),
                types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event")
            )

            BOT.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=f"📅 DateTime set to: {selected_dt.strftime('%Y-%m-%d %H:%M')}\n\n"
            )

            BOT.send_message(
                chat_id,
                "⏳ How long will it last? (Default: 1 hour)",
                reply_markup=markup
            )
        else:
            BOT.send_message(chat_id, "Time selection cancelled")
            BOT.delete_message(chat_id, message_id)

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
        f"Time: {event['datetime']}\n"
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
            # Add event to calendar
            result = add_ical_event(
                event_name=event_data[chat_id]['name'],
                start_time=event_data[chat_id]['datetime'],
                duration_hours=event_data[chat_id]['duration'],
                description=event_data[chat_id].get('description', ''),
                location=event_data[chat_id].get('location', '')
            )
            
            BOT.send_message(chat_id, result)
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

########################## Authorisation denied ##################################
@BOT.message_handler(commands=['addevent'])
def deny_access_add(message):
    BOT.reply_to(message, "Access denied: You are not authorized.")

@BOT.message_handler(commands=['deleteEvent'])
def deny_access_delete(message):
    BOT.reply_to(message, "Access denied: You are not authorized.")
##################################################################################



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
        BOT.reply_to(message, f"🚫 Unexpected error: You have no more free API credits !")
        sending.stop()
        s.join()


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

        response = client.chat.completions.create(
            model="n/a",
            messages=conversation_history[chat_id]
        )

        # Add assistant response to history
        response_text = response.choices[0].message.content.strip()
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

if __name__ == "__main__":
    client = OpenAI(
        base_url=agent_endpoint,
        api_key=agent_access_key
    )

    # Start in a separate thread for better control
    bot_thread = Thread(target=run_bot, daemon=True)
    bot_thread.start()

    print("Starting bot online!")
    
    # Keep main thread alive
    while True:
        time.sleep(1)