from telebot import types
import datetime
from typing import Dict, Optional, Tuple

class TimePicker:
    def __init__(self):
        self.user_selections: Dict[int, datetime.datetime] = {}  # Store datetime objects instead of time

    def clear_selection(self, chat_id: int):
        """Clear any stored time selection for this chat"""
        self.user_selections.pop(chat_id, None)
        
    def create_time_picker(self, chat_id: int, base_date: datetime.date) -> types.InlineKeyboardMarkup:
        """Create time picker attached to a specific date"""
        if chat_id not in self.user_selections:
            # Default to noon on the selected date
            self.user_selections[chat_id] = datetime.datetime.combine(
                base_date,
                datetime.time(12, 0)
            )
        
        current_dt = self.user_selections[chat_id]
        
        markup = types.InlineKeyboardMarkup(row_width=5)
        
        # Time display header
        time_display = f"⏰ {current_dt.strftime('%I:%M %p')} ⏰"
        markup.row(types.InlineKeyboardButton(time_display, callback_data="time_display"))
        
        # Hour controls
        hour_controls = [
            types.InlineKeyboardButton("◀️", callback_data="time_hour_decr"),
            types.InlineKeyboardButton("-1", callback_data="time_hour_decr1"),
            types.InlineKeyboardButton(f"{current_dt.hour:02d}", callback_data="time_hour_display"),
            types.InlineKeyboardButton("+1", callback_data="time_hour_incr1"),
            types.InlineKeyboardButton("▶️", callback_data="time_hour_incr")
        ]
        markup.row(*hour_controls)
        
        # Minute controls
        minute_controls = [
            types.InlineKeyboardButton("◀️", callback_data="time_minute_decr"),
            types.InlineKeyboardButton("-5", callback_data="time_minute_decr5"),
            types.InlineKeyboardButton(f"{current_dt.minute:02d}", callback_data="time_minute_display"),
            types.InlineKeyboardButton("+5", callback_data="time_minute_incr5"),
            types.InlineKeyboardButton("▶️", callback_data="time_minute_incr")
        ]
        markup.row(*minute_controls)
        
        # AM/PM toggle
        ampm = "🌞 AM" if current_dt.hour < 12 else "🌙 PM"
        markup.row(types.InlineKeyboardButton(ampm, callback_data="time_toggle_ampm"))
        
        # # Quick time presets
        # presets = [
        #     ("🌅 Morning", 9),
        #     ("☀️ Noon", 12),
        #     ("🌇 Afternoon", 15),
        #     ("🌃 Evening", 18),
        #     ("🌜 Night", 21)
        # ]
        # markup.row(*[
        #     types.InlineKeyboardButton(text, callback_data=f"time_preset_{hour}")
        #     for text, hour in presets
        # ])
        
        # Action buttons
        markup.row(
            types.InlineKeyboardButton("✅ Confirm Time", callback_data="time_confirm"),
            types.InlineKeyboardButton("❌ Cancel", callback_data="time_cancel")
        )
        
        return markup
    
    def handle_callback(self, call, base_date: datetime.date) -> Tuple[bool, Optional[datetime.datetime]]:
        """Process time picker callbacks"""
        chat_id = call.message.chat.id
        
        # Initialize if not exists
        if chat_id not in self.user_selections:
            self.user_selections[chat_id] = datetime.datetime.combine(
                base_date,
                datetime.time(12, 0)
            )
        
        current_dt = self.user_selections[chat_id]
        
        try:
            data = call.data
            
            if data == "time_hour_incr":
                new_hour = (current_dt.hour + 1) % 24
                new_dt = current_dt.replace(hour=new_hour)
            
            elif data == "time_hour_incr1":
                new_hour = (current_dt.hour + 1) % 24
                new_dt = current_dt.replace(hour=new_hour)
            
            elif data == "time_hour_decr":
                new_hour = (current_dt.hour - 1) % 24
                new_dt = current_dt.replace(hour=new_hour)
            
            elif data == "time_hour_decr1":
                new_hour = (current_dt.hour - 1) % 24
                new_dt = current_dt.replace(hour=new_hour)
            
            elif data == "time_minute_incr":
                new_minute = (current_dt.minute + 1) % 60
                new_dt = current_dt.replace(minute=new_minute)
            
            elif data == "time_minute_incr5":
                new_minute = (current_dt.minute + 5) % 60
                new_dt = current_dt.replace(minute=new_minute)
            
            elif data == "time_minute_decr":
                new_minute = (current_dt.minute - 1) % 60
                new_dt = current_dt.replace(minute=new_minute)
            
            elif data == "time_minute_decr5":
                new_minute = (current_dt.minute - 5) % 60
                new_dt = current_dt.replace(minute=new_minute)
            
            elif data == "time_toggle_ampm":
                new_hour = (current_dt.hour + 12) % 24
                new_dt = current_dt.replace(hour=new_hour)
            
            elif data.startswith("time_preset_"):
                hour = int(data.split("_")[1])
                new_dt = current_dt.replace(hour=hour, minute=0)
            
            elif data == "time_confirm":
                return (False, current_dt)
            
            elif data == "time_cancel":
                return (False, None)
            
            else:
                return (True, None)
            
            # Update the stored datetime
            self.user_selections[chat_id] = new_dt
            return (True, None)
            
        except Exception as e:
            print(f"Error handling time picker: {e}")
            return (False, None)