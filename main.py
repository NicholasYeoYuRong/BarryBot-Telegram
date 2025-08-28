from dotenv import load_dotenv
from threading import Thread
from collections import defaultdict
from io import BytesIO
import pytz
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
from redis_database import save_user, delete_user, get_user_username, get_all_subscribed_chats, is_subscribed, get_user_chat_id, save_user_to_database, get_all_user_usernames
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import googlemaps
import math


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

ALLOWED_USERS = ["Nicholas_yowo"]
OWNER = "Nicholas_yowo"

BOT = telebot.TeleBot(token=API_TOKEN)
time_picker = TimePicker()

scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Singapore"))
scheduler.start()

# Register a shutdown hook to stop the scheduler gracefully
atexit.register(lambda: scheduler.shutdown())

# STORING CONVO HISTORY
conversation_history = defaultdict(list)

# STORING USER LOCATION
user_locations = {}

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else None

def get_nearby_food_places(latitude, longitude, radius=1000, food_type=None):
    """
    Get nearby food places using Google Places API
    """
    if not gmaps:
        return None, "Google Maps API not configured"
    
    try:
        # Build the request
        places_result = gmaps.places_nearby(
            location=(latitude, longitude),
            radius=radius,
            type='restaurant',
            keyword=food_type if food_type else None,
            open_now=True  # Only show places currently open
        )
        
        places = places_result.get('results', [])
        
        if not places:
            return [], "No food places found nearby"
        
        # Sort by rating (highest first)
        places.sort(key=lambda x: x.get('rating', 0), reverse=True)
        
        # Get detailed information for top 20 places
        top_places = []
        for place in places[:20]:
            place_details = gmaps.place(place['place_id'])
            detailed_info = place_details.get('result', {})
            
            top_places.append({
                'name': place.get('name', 'Unknown'),
                'rating': place.get('rating', 'No rating'),
                'price_level': place.get('price_level', 'Unknown'),
                'vicinity': place.get('vicinity', 'No address'),
                'types': place.get('types', []),
                'opening_hours': detailed_info.get('opening_hours', {}).get('weekday_text', ['Hours not available']),
                'phone': detailed_info.get('formatted_phone_number', 'No phone'),
                'website': detailed_info.get('website', 'No website'),
                'location': place['geometry']['location'],
                'place_id': place['place_id']
            })
        
        return top_places, None
        
    except Exception as e:
        return None, f"Error fetching places: {str(e)}"
    
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in meters"""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi/2) * math.sin(delta_phi/2) +
         math.cos(phi1) * math.cos(phi2) *
         math.sin(delta_lambda/2) * math.sin(delta_lambda/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c
    
def format_place_message(place, user_lat, user_lon):
    """Format a single place into a readable message"""
    distance = calculate_distance(user_lat, user_lon, 
                                place['location']['lat'], 
                                place['location']['lng'])
    
    # Price level emoji mapping
    price_emojis = {
        0: '💰',  # Free
        1: '💵',  # Inexpensive
        2: '💵💵',  # Moderate
        3: '💵💵💵',  # Expensive
        4: '💵💵💵💵'  # Very Expensive
    }
    
    price_display = price_emojis.get(place.get('price_level', 0), '💰')
    
    message = (
        f"🍽️ **{place['name']}**\n"
        f"⭐ Rating: {place['rating']}/5\n"
        f"💰 Price: {price_display}\n"
        f"📍 Distance: {distance:.0f}m away\n"
        f"🏠 Address: {place['vicinity']}\n"
    )
    
    if place.get('phone') != 'No phone':
        message += f"📞 Phone: {place['phone']}\n"
    
    # Add Google Maps link
    # maps_link = f"https://www.google.com/maps/place/?q=place_id:{place['place_id']}"
    # message += f"🗺️ [View on Google Maps]({maps_link})"
    
    return message

# Restore scheduled jobs
def restore_scheduled_jobs():
    for chat_id in get_all_subscribed_chats():
        job_id = f"positive_msg_{chat_id}"
        if not scheduler.get_job(job_id):
            scheduler.add_job(
                lambda chat_id=chat_id: positive_message(chat_id),
                'cron',
                hour=8,
                minute=0,
                id=job_id
            )

restore_scheduled_jobs()

def stop_all_positive_message_jobs():
    """Stop all positive message jobs for all subscribed users."""
    for chat_id in get_all_subscribed_chats():
        job_id = f"positive_msg_{chat_id}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)

# Positive message job
def positive_message(chat_id):
    """Send a daily positive message to the user."""
    try:

        conversation_history[chat_id] = conversation_history[chat_id][-6:]

        system_content = "You are a helpful assistant. Keep responses concise."
        message_content = f"""Greet me based on the time of the day and give me a different positive message. Add a quote and affirmation from the bible as to what God wants to tell me today. Use emoji just for this response. Go by this format:
        Good morning [{get_user_username(chat_id) or 'friend'}]!\n\n
        TODAY'S POSITIVE MESSAGE: [positive message]\n
        WHAT GOD IS TELLING YOU TODAY: [quote] [Bible verse]\n
        TODAY'S AFFIRMATION: [affirmation]"""

        if chat_id not in conversation_history:
            conversation_history[chat_id] = [
                {'role': 'system', 'content': system_content}
            ]

        # Add user message to history
        conversation_history[chat_id].append(
            {'role': 'user', 'content': message_content}
        )

        response = client.chat.completions.create(
            model="n/a",
            messages=conversation_history[chat_id],
            temperature=0.9
        )

        ## Add assistant's response to conversation history ##
        response_text = response.choices[0].message.content.strip()
        conversation_history[chat_id].append(
            {'role': 'assistant', 'content': response_text}
        )

        BOT.send_message(chat_id, response_text)

    except Exception as e:
        print(f"Failed to send to {chat_id}: {e}")
        delete_user(chat_id)
        scheduler.remove_job(f"positive_msg_{chat_id}")

################## Check if user is authorized ###################
def is_allowed_user(message: types.Message) -> bool:
    return message.from_user.username in ALLOWED_USERS

def is_owner(message: types.Message) -> bool:
    return message.from_user.username == OWNER
###################################################################

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
    else:
        welcome_text = f'Hi {message.from_user.first_name}, My name is Barry! How can I assist you today?'
        BOT.send_message(message.chat.id, welcome_text)
    
    save_user_to_database(message.chat.id, message.from_user.username)
    
    BOT.send_message(message.chat.id, "Type /help to see available commands.")

@BOT.message_handler(commands=['setlocation'])
def request_location(message):
    """Ask user to share their location"""
    markup = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    location_btn = types.KeyboardButton("📍 Share Location", request_location=True)
    markup.add(location_btn)
    
    BOT.send_message(
        message.chat.id,
        "Please share your location so I will be able to help your better:",
        reply_markup=markup
    )    

@BOT.message_handler(content_types=['location'])
def handle_location(message):
    """Store User's location"""
    chat_id = message.chat.id
    location = message.location
    user_locations[chat_id] = (location.latitude, location.longitude)

    BOT.send_message(
        chat_id,
        f"📍 Location saved!\n",
        reply_markup=types.ReplyKeyboardRemove()
    )

@BOT.message_handler(commands=['food', 'restaurant', 'eat'])
def find_food_places(message):
    """Find nearby food places"""
    chat_id = message.chat.id

    if chat_id not in user_locations:
        BOT.send_message(
            chat_id,
            "📍 I need your location first! Please use /setlocation to share your location."
        )
        return
        # request_location(message)

    radius = 200
    food_type = None

    user_lat, user_lon = user_locations[chat_id]
    
    places, error = get_nearby_food_places(user_lat, user_lon, radius, food_type)
    
    if error:
        BOT.send_message(chat_id, f"❌ {error}")
        return
    
    if not places:
        BOT.send_message(chat_id, "🍽️ No food places found nearby.")
        return
    
    # Send ALL places
    if len(places) > 0:
        all_places = "\n\n".join([
            format_place_message(place, user_lat, user_lon) 
            for place in places[0:20]  # Show next 20 places
        ])
        
        BOT.send_message(
            chat_id,
            f"🍽️ **ALL NEARBY OPTIONS**\n\n{all_places}",
            parse_mode='Markdown',
            disable_web_page_preview=True
        )
    

@BOT.message_handler(commands=['help'])
def help_command(message):
    help_text = (
        "Here are the commands you can use:\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/subscribe - Subscribe to daily positive messages\n"
        "/unsubscribe - Unsubscribe from daily positive messages\n"
        "/addschedule - Add an event to your calendar (Only for authorized users)\n"
        "/deleteschedule - Delete an event from your calendar (Only for authorized users)\n"
        "/upcomingschedule - List your upcoming schedules\n"
        "/allschedule - List all your schedules\n"
        "/reset - Reset the chat history\n"
        "/food - Find nearby food places\n"
        "/setlocation - Set your location\n"
    )
    BOT.send_message(message.chat.id, help_text)

@BOT.message_handler(commands=['getalluser'])
def getalluser(message):
    usernames = get_all_user_usernames()
    if usernames:
        BOT.send_message(message.chat.id, "All users:\n" + "\n".join(usernames))
    else:
        BOT.send_message(message.chat.id, "No users found.")

@BOT.message_handler(commands=['reset'])
def reset_chat(message):
    conversation_history[message.chat.id] = []
    BOT.send_message(message.chat.id, "***CHAT RESETTED***")

@BOT.message_handler(commands=['restartscheduler'], func=is_owner)
def restart_scheduler(message):
    stop_all_positive_message_jobs()
    restore_scheduled_jobs()
    BOT.send_message(message.chat.id, "***SCHEDULER RESTARTED***")

@BOT.message_handler(commands=['upcomingschedule'])
def list_calendar(message):
    try:

        events = get_ical_events(ICAL_URL)
        current_time = datetime.now().astimezone()

        future_events = []
        for event_text in events:
            event_time_str = extract_datetime(event_text)
            event_time = datetime.fromisoformat(event_time_str).astimezone()

            if event_time >= current_time:
                future_events.append((event_time, event_text))

        future_events.sort()

        print(f"Total events from calendar: {len(events)}")
        print(f"Future events after filtering: {len(future_events)}")

        formatted_events = []
        if not future_events:
            formatted_events.append("Your schedule is free! There are no upcoming events!")
        else:
            for i, (event_time, event_text) in enumerate(future_events[:5], 1):
                formatted_events.append(
                    f"SCHEDULE {i}:\n"
                    f"  {format_event(event_text)}")

        structured_events = "\n\n".join(formatted_events)
        BOT.send_message(message.chat.id, f"===YOUR UPCOMING SCHEDULES===\n\n{structured_events}")
    
    except Exception as e:
        print(f"🚫 Unexpected error: {str(e)}")

@BOT.message_handler(commands=['allschedule'])
def list_calendar(message):
    try:

        events = get_ical_events(ICAL_URL)
        current_time = datetime.now().astimezone()

        future_events = []
        for event_text in events:
            event_time_str = extract_datetime(event_text)
            event_time = datetime.fromisoformat(event_time_str).astimezone()

            if event_time >= current_time:
                future_events.append((event_time, event_text))

        future_events.sort()

        print(f"Total events from calendar: {len(events)}")
        print(f"Future events after filtering: {len(future_events)}")

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
@BOT.message_handler(commands=['deleteschedule'], func=is_allowed_user)
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
            "=========DELETE AN EVENT=======\n\n"
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
        text="Deletion cancelled. You can start over with /deleteschedule"
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
@BOT.message_handler(commands=['addschedule'], func=is_allowed_user)
def start_add_event(message):
    chat_id = message.chat.id
    user_states[chat_id] = 'awaiting_event_name'
    event_data[chat_id] = {
        'name': None,
        'datetime': None,
        'date_only': None,
        'duration': 1.0,  # Default duration
        'description': None,
        'location': None
    }  # Set default duration
    time_picker.clear_selection(chat_id) # Clear any previous time selection

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
            user_states.pop(chat_id, None)
            event_data.pop(chat_id, None)
            time_picker.clear_selection(chat_id)
            BOT.send_message(chat_id, "Time selection cancelled")
            BOT.delete_message(chat_id, message_id)

@BOT.callback_query_handler(func=lambda call: call.data == "skip_duration")
def skip_duration(call):
    chat_id = call.message.chat.id
    user_states[chat_id] = 'awaiting_description'
    
    # Create inline skip button
    markup = types.InlineKeyboardMarkup()
    markup.add(
        types.InlineKeyboardButton("Skip description", callback_data="skip_description"),
        types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event")
    )

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
    markup.add(
        types.InlineKeyboardButton("Skip description", callback_data="skip_description"),
        types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event")
    )

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
    markup.add(
        types.InlineKeyboardButton("Skip location", callback_data="skip_location"),
        types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event")
    )
    
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
    markup.add(
        types.InlineKeyboardButton("Skip location", callback_data="skip_location"),
        types.InlineKeyboardButton("Cancel", callback_data="cancel_add_event")
    )
    
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
        BOT.send_message(chat_id, "Event cancelled. Start over with /addschedule")
    
    # Clean up
    user_states.pop(chat_id, None)
    event_data.pop(chat_id, None)
    BOT.delete_message(chat_id, call.message.message_id)

@BOT.callback_query_handler(func=lambda call: call.data == "cancel_add_event")
def cancel_add_event(call):
    chat_id = call.message.chat.id

    # Clean up user state and event data
    user_states.pop(chat_id, None)
    event_data.pop(chat_id, None)
    time_picker.clear_selection(chat_id)
    
    BOT.edit_message_text(
        chat_id=chat_id,
        message_id=call.message.message_id,
        text="Event creation cancelled. You can start over with /addschedule"
    )

########################## Authorisation denied ##################################
@BOT.message_handler(commands=['addschedule'])
def deny_access_add(message):
    BOT.reply_to(message, "Access denied: You are not authorized.")

@BOT.message_handler(commands=['deleteschedule'])
def deny_access_delete(message):
    BOT.reply_to(message, "Access denied: You are not authorized.")

@BOT.message_handler(commands=['restartscheduler'])
def deny_access_restart(message):
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

## SUBSCRIBE TO DAILY POSITIVE MESSAGES ##
@BOT.message_handler(commands=['subscribe'])
def subscribe(message):
    chat_id = message.chat.id
    username = message.from_user.username

    if is_subscribed(chat_id):
        BOT.reply_to(
            message, 
            "You are already subscribed to daily positive messages! 🌟\n"
            "You can unsubscribe at any time by sending /unsubscribe."
        )
        return

    # Store user in database
    save_user(chat_id, username)

    # Create unique job ID for this user
    job_id = f"positive_msg_{chat_id}"
    if not scheduler.get_job(job_id):
        scheduler.add_job(
            lambda chat_id=chat_id: positive_message(chat_id),  # Wrapped in lambda
            'cron',
            hour=8,
            minute=0,
            id=job_id,
            replace_existing=True
        )

    BOT.send_message(
        chat_id, 
        "You will receive daily positive messages at 8 am! 🌟\n"
        "You can unsubscribe at any time by sending /unsubscribe."
    )
    

## UNSUBSCRIBE FROM DAILY POSITIVE MESSAGES ##
@BOT.message_handler(commands=['unsubscribe'])
def unsubscribe(message):
    chat_id = message.chat.id

    if not is_subscribed(chat_id):
        BOT.send_message(chat_id, "You're not currently subscribed.")
        return

    # Delete user from database
    delete_user(chat_id)
    scheduler.remove_job(f"positive_msg_{chat_id}")

    BOT.send_message(
        chat_id,
        "🔕 You've been unsubscribed.\n"
        "Use /subscribe to restart messages."
    )


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
        conversation_history[chat_id] = conversation_history[chat_id][-6:]
        if chat_id not in conversation_history:
            conversation_history[chat_id] = [
                {'role': 'system', 'content': "You are a helpful assistant. Keep responses concise. Use emojis in responses."}
            ]

        # Add user message to history
        conversation_history[chat_id].append(
            {'role': 'user', 'content': message.text}
        )

        full_response = ""
        while True:
            response = client.chat.completions.create(
                model="n/a",
                messages=conversation_history[chat_id],
                max_tokens=4000, # Limit response length
            )
            chunk = response.choices[0].message.content.strip()
            full_response += chunk

            if response.choices[0].finish_reason != 'length':
                break

        # Add assistant response to history
        response_text = full_response.strip()

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