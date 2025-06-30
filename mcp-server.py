from config import ICAL_URL, ICLOUD_USERNAME, ICLOUD_PASSWORD, CALDAV_URL, CALENDAR_NAME
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta, time
from ics import Calendar
from caldav import DAVClient
import re
from dateutil import parser
import pytz
from zoneinfo import ZoneInfo
import tzlocal

mcp = FastMCP(
    name="MCP-server",
    host="0.0.0.0",
    port="8000",
    )
    
LOCAL_TIMEZONE = pytz.timezone('Asia/Singapore')

def parse_natural_time(time_str: str) -> datetime:
    """Parse natural language time into datetime with comprehensive format support"""
    try:
        # Clean input and prepare
        input_str = time_str.lower().strip()
        now = datetime.now(LOCAL_TIMEZONE)
        base_date = now.date()
        
        # 1. Handle exact ISO format (2023-12-25 14:30)
        if re.match(r'^\d{4}-\d{2}-\d{2} \d{1,2}(?::\d{2})?\s*(am|pm)?$', time_str, re.IGNORECASE):
            # Split into date and time parts
            date_section, time_section = time_str.split(' ', 1)

            # Parse the date portion
            dt = datetime.strptime(date_section, '%Y-%m-%d')

            # Parse time portion
            time_match = re.match(
                r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
                time_section,
                re.IGNORECASE
            )

            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                period = time_match.group(3)

                if period and period.lower() == 'pm' and hour < 12:
                    hour += 12
                elif period and period.lower() == 'am' and hour == 12:
                    hour = 0

                dt = dt.replace(hour=hour, minute=minute)
                return LOCAL_TIMEZONE.localize(dt)
        
        # 2. Handle "tomorrow" cases (Tomorrow 2pm, Tomorrow 2:30pm)
        if "tomorrow" in input_str:
            base_date += timedelta(days=1)
            time_part = input_str.replace("tomorrow", "").strip()
            if not time_part:  # Just "tomorrow"
                return LOCAL_TIMEZONE.localize(datetime.combine(base_date, time(0, 0)))
            input_str = time_part
        
        # 3. Handle "next [weekday]" cases (Next Friday 3pm)
        next_match = re.match(r'next\s+(\w+)\b', input_str)
        if next_match:
            weekday = next_match.group(1)
            days_ahead = (_weekday_to_num(weekday) - now.weekday() + 7) % 7 or 7
            base_date += timedelta(days=days_ahead)
            input_str = input_str.replace(next_match.group(0), "").strip()
        
        # 4. Handle standalone weekdays (Sunday 5:30pm)
        weekday_match = re.match(r'(\w+)\b', input_str)
        if weekday_match and weekday_match.group(1) in _weekday_names():
            weekday = weekday_match.group(1)
            days_ahead = (_weekday_to_num(weekday) - now.weekday()) % 7
            if days_ahead == 0 and now.time() > _parse_time_part(input_str):
                days_ahead = 7
            base_date += timedelta(days=days_ahead)
            input_str = input_str.replace(weekday_match.group(0), "").strip()
        
        # 5. Parse time component
        time_obj = _parse_time_part(input_str)
        
        # Combine date and time
        result = LOCAL_TIMEZONE.localize(datetime.combine(base_date, time_obj))
        
        # If no specific date mentioned and time is in past, move to next day
        if not _has_date_keyword(time_str):
            if result < now:
                result += timedelta(days=1)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Could not parse time: {time_str}") from e

# Helper functions
def _weekday_names():
    return ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

def _weekday_to_num(weekday: str) -> int:
    """Convert weekday name to number (0=Monday, 6=Sunday)"""
    return _weekday_names().index(weekday.lower())

def _parse_time_part(time_str: str) -> time:
    """Parse just the time portion from a string"""
    time_match = re.search(
        r'(?P<hour>\d{1,2})'
        r'(?:[.:](?P<minute>\d{2}))?'
        r'\s*'
        r'(?P<period>am|pm)?',
        time_str.strip(),
        re.IGNORECASE
    )
    
    if not time_match:
        return time(0, 0)  # Default to midnight
    
    hour = int(time_match.group('hour'))
    minute = int(time_match.group('minute')) if time_match.group('minute') else 0
    period = time_match.group('period')
    
    # Convert to 24-hour format
    if period:
        if period.lower() == 'pm' and hour < 12:
            hour += 12
        elif period.lower() == 'am' and hour == 12:
            hour = 0
    
    # Validate time
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time: {hour}:{minute}")
    
    return time(hour, minute)

def _has_date_keyword(time_str: str) -> bool:
    """Check if string contains date-related keywords"""
    keywords = ['tomorrow', 'next'] + _weekday_names()
    return any(keyword in time_str.lower() for keyword in keywords)

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

# Tool for retrieving events from calendar
@mcp.tool()
def get_ical_events(ical_url: str, max_results: int = 10) -> list[str]:
    """Fetch events from an iCal feed URL."""
    response = httpx.get(ical_url)
    calendar = Calendar(response.text)

    events = []
    for e in list(calendar.events)[:max_results]:
        events.append(
            f"{e.name} | "
            f"{e.duration if e.duration else 'None'} | "
            f"{e.description if e.description else 'None'} | "
            f"{e.location if e.location else 'None'} | "
            f"{e.begin}"
        )

    return events

@mcp.tool()
def get_all_ical_events(ical_url: str) -> list[str]:
    """Fetch ALL events from an iCal feed URL"""
    response = httpx.get(ical_url)
    calendar = Calendar(response.text)

    events = []
    for e in list(calendar.events):
        events.append(f"{e.name} | {e.begin}") ## CHANGES TO " | "
    
    return events

# Tool for adding events to calendar #
@mcp.tool()
def add_ical_event(
    event_name: str,
    start_time: str,
    duration_hours: float = 1.0,
    description: str = "",
    location: str = ""
) -> str:
    """Add event to iCloud calendar"""
    try:
        start_dt = parse_natural_time(start_time)
        end_dt = start_dt + timedelta(hours=duration_hours)

        #Connect to iCloud via CalDAV
        with DAVClient(
            url=CALDAV_URL,
            username=ICLOUD_USERNAME,
            password=ICLOUD_PASSWORD
        ) as client:
            principal = client.principal()
            calendar = principal.calendar(name=CALENDAR_NAME)

            # Create the event
            calendar.save_event(
                f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:{int(start_dt.timestamp())}@telegram-bot
DTSTAMP:{datetime.now(pytz.UTC).strftime("%Y%m%dT%H%M%SZ")}
DTSTART;TZID={LOCAL_TIMEZONE.zone}:{start_dt.strftime("%Y%m%dT%H%M%SZ")}
DTEND;TZID={LOCAL_TIMEZONE.zone}:{end_dt.strftime("%Y%m%dT%H%M%SZ")}
SUMMARY:{event_name}
DESCRIPTION:{description or ''}
LOCATION:{location or ''}
END:VEVENT
END:VCALENDAR"""
            )
            
            return (f"✅ ADDED TO CALENDAR: {CALENDAR_NAME}:\n\n"
                   f"- 📝 EVENT: {event_name}\n"
                   f"- ⏰ TIME: {start_dt.strftime('%a %b %d at %I:%M %p')}\n"
                   f"- ⏳ DURATION: {duration_hours} hours\n"
                   f"- 📍 LOCATION: {location or 'No location specified'}")
        
    except Exception as e:
        return f"❌ Failed to add event: {str(e)}"
            
# Tool for deleting events from calendar
@mcp.tool()
def delete_calendar_event(event_name: str, event_time: str) -> str:
    """Delete event from iCloud calendar"""
    try:
        event_dt = datetime.fromisoformat(event_time)

        with DAVClient(
            url=CALDAV_URL,
            username=ICLOUD_USERNAME,
            password=ICLOUD_PASSWORD,
        ) as client:
            principal = client.principal()
            calendar = principal.calendar(name=CALENDAR_NAME)
            
            # Find matching events
            events = calendar.search(
                start=event_dt - timedelta(hours=1),
                end=event_dt + timedelta(hours=1),
            )

            for event in events:
                if (event_name == event.vobject_instance.vevent.summary.value and
                    event_dt == event.vobject_instance.vevent.dtstart.value):
                    event.delete()
                    return f"✅ Deleted: {event_name} @ {event_dt.strftime('%Y-%m-%d %H:%M')}"
            
            return f"❌ Event not found: {event_name} @ {event_dt.strftime('%Y-%m-%d %H:%M')}"
        
    except Exception as e:
        return f"❌ Failed to delete event: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
