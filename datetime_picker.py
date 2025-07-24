from telebot import types
import datetime
from typing import Optional, Tuple, Union

class DateTimePicker:
    def __init__(self):
        self.user_selections = {}  # Stores {chat_id: {'datetime': datetime, 'message_id': int}}

    def create_datetime_picker(self, selected_datetime: Optional[datetime.datetime] = None) -> types.InlineKeyboardMarkup:
        """Create interactive datetime picker"""
        if selected_datetime is None:
            selected_datetime = datetime.datetime.now()
        
        markup = types.InlineKeyboardMarkup(row_width=7)
        
        # Month/year header
        markup.row(
            types.InlineKeyboardButton("◀️", callback_data="dt_prev_month"),
            types.InlineKeyboardButton(selected_datetime.strftime("%B %Y"), callback_data="dt_month_year"),
            types.InlineKeyboardButton("▶️", callback_data="dt_next_month")
        )
        
        # Weekday headers
        markup.row(*[types.InlineKeyboardButton(day, callback_data="dt_ignore") 
                   for day in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]])
        
        # Calendar days
        first_day = selected_datetime.replace(day=1)
        starting_weekday = first_day.weekday()
        days_in_month = (first_day.replace(month=first_day.month % 12 + 1, day=1) - datetime.timedelta(days=1)).day
        
        days = [""] * starting_weekday
        days.extend([f"[{d}]" if d == selected_datetime.day else str(d) for d in range(1, days_in_month + 1)])
        
        for i in range(0, len(days), 7):
            markup.row(*[
                types.InlineKeyboardButton(
                    day.strip("[]"),
                    callback_data=f"dt_day_{day.strip('[]')}" if day else "dt_ignore"
                ) for day in days[i:i+7]
            ])
        
        # Time controls
        markup.row(
            types.InlineKeyboardButton("🕒", callback_data="dt_ignore"),
            types.InlineKeyboardButton("⬇️", callback_data="dt_hour_decr"),
            types.InlineKeyboardButton(f"{selected_datetime.hour:02d}", callback_data="dt_ignore"),
            types.InlineKeyboardButton("⬆️", callback_data="dt_hour_incr"),
            types.InlineKeyboardButton(":", callback_data="dt_ignore"),
            types.InlineKeyboardButton("⬇️", callback_data="dt_minute_decr"),
            types.InlineKeyboardButton(f"{selected_datetime.minute:02d}", callback_data="dt_ignore"),
            types.InlineKeyboardButton("⬆️", callback_data="dt_minute_incr")
        )
        
        # Action buttons
        markup.row(
            types.InlineKeyboardButton("✅ Confirm", callback_data="dt_confirm"),
            types.InlineKeyboardButton("❌ Cancel", callback_data="dt_cancel")
        )
        
        return markup

    def handle_callback(self, call) -> Tuple[bool, Union[str, None]]:
        """Handle picker callbacks and return formatted datetime string"""
        chat_id = call.message.chat.id
        data = call.data
        
        # Get current selection or initialize
        if chat_id not in self.user_selections:
            self.user_selections[chat_id] = {
                'datetime': datetime.datetime.now(),
                'message_id': call.message.message_id
            }
        
        current_dt = self.user_selections[chat_id]['datetime']
        
        try:
            if data == "dt_prev_month":
                new_date = (current_dt.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
                self.user_selections[chat_id]['datetime'] = new_date
                return (True, None)
                
            elif data == "dt_next_month":
                next_month = current_dt.month % 12 + 1
                next_year = current_dt.year + (1 if next_month == 1 else 0)
                new_date = current_dt.replace(month=next_month, year=next_year, day=1)
                self.user_selections[chat_id]['datetime'] = new_date
                return (True, None)
                
            elif data.startswith("dt_day_"):
                day = int(data.split("_")[2])
                self.user_selections[chat_id]['datetime'] = current_dt.replace(day=day)
                return (True, None)
                
            elif data in ["dt_hour_incr", "dt_hour_decr"]:
                change = 1 if data == "dt_hour_incr" else -1
                new_hour = (current_dt.hour + change) % 24
                self.user_selections[chat_id]['datetime'] = current_dt.replace(hour=new_hour)
                return (True, None)
                
            elif data in ["dt_minute_incr", "dt_minute_decr"]:
                change = 1 if data == "dt_minute_incr" else -1
                new_minute = (current_dt.minute + change) % 60
                self.user_selections[chat_id]['datetime'] = current_dt.replace(minute=new_minute)
                return (True, None)
                
            elif data == "dt_confirm":
                selected_dt = self.user_selections[chat_id]['datetime']
                # Return formatted datetime string
                return (False, selected_dt.strftime("%Y-%m-%d %H:%M"))
                
            elif data == "dt_cancel":
                return (False, None)
                
            return (True, None)
        except Exception as e:
            print(f"Error handling datetime picker: {e}")
            return (False, None)