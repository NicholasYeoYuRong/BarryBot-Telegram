from config import ICAL_URL, ICLOUD_USERNAME, ICLOUD_PASSWORD, CALDAV_URL, CALENDAR_NAME
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta
from ics import Calendar
from caldav import DAVClient
import re
from dateutil import parser

mcp = FastMCP(
    name="MCP-server",
    host="0.0.0.0",
    port="8000",
    )

def parse_natural_time(time_str: str) -> datetime:
    """Parse natural language time into datetime"""

    try:

        if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$', time_str): ## 1. "^" : Start of string  2. "\d" : Any single digit (0-9) 3. "$" : End of string
            return datetime.strptime(time_str, '%Y-%m-%d %H:%M')
        
        # Parse natural language time (e.g. "tomorrow at 5pm" -> datetime object for tomorrow, 17:00) #
        dt = parser.parse(time_str, fuzzy=True)

        # If date is TODAY and "tomorrow" is not in the input" #
        if dt.date == datetime.now().date() and "tomorrow" not in time_str.lower():
            # If time is in past #
            if dt.time() < datetime.now().time():
                # This will assume user meant tomorrow, and adds 1 day #
                # Example: if someone writes "4pm" and it's already 6pm, we probably want to interpret it as 4pm tomorrow, not earlier today.
                dt += timedelta(days=1)

        return dt
    except Exception as e:
        raise ValueError(f"Could not parse time: {time_str}") from e

@mcp.tool()
def list_task(max_results: int) -> list[str]:
    """List all tasks"""
    return[
        "East breakfast",
        "Go to the gym",
        "Read a book",
    ][:max_results]

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

@mcp.tool()
def get_ical_events(ical_url: str, max_results: int = 10) -> list[str]:
    """Fetch events from an iCal feed URL."""
    response = httpx.get(ical_url)
    calendar = Calendar(response.text)

    events = []
    for e in list(calendar.events)[:max_results]:
        events.append(f"{e.name} at {e.begin}")

    return events

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
DTSTAMP:{datetime.now().strftime("%Y%m%dT%H%M%SZ")}
DTSTART:{start_dt.strftime("%Y%m%dT%H%M%SZ")}
DTEND:{end_dt.strftime("%Y%m%dT%H%M%SZ")}
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
            

if __name__ == "__main__":
    mcp.run(transport="stdio")
