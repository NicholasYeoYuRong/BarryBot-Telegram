import googlemaps
import math
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else None

def get_nearby_food_places(latitude, longitude, radius=400, food_type=None):
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
        
        # Get detailed information for top 15 places
        top_places = []
        for place in places[:15]:
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
        f"🍽️ <b>{place.get('name', 'Unknown')}</b>\n"
        f"⭐ Rating: {place.get('rating', '?')}/5\n"
        f"💰 Price: {price_display}\n"
        f"📍 Distance: {distance:.0f}m away\n"
        f"🏠 Address: {place.get('vicinity', 'No address')}\n"
    )
    
    if place.get('phone') != 'No phone':
        message += f"📞 Phone: {place.get('phone', 'No phone')}\n"

    # Add Google Maps link
    maps_link = f"https://www.google.com/maps/search/?api=1&query={place.get('name', 'Unknown')}&query_place_id={place.get('place_id', '')}"
    message += f'🗺️ <a href="{maps_link}">View on Google Maps</a>'
    
    return message

def get_location_name(latitude, longitude):
    """
    Get human-readable location name from coordinates using Google Maps Geocoding API
    """
    if not gmaps:
        return f"{latitude:.6f}, {longitude:.6f}"  # Fallback to coordinates
    
    try:
        # Reverse geocode the coordinates
        reverse_geocode_result = gmaps.reverse_geocode((latitude, longitude))
        
        if reverse_geocode_result:
            # Extract the formatted address (most human-readable)
            location_name = reverse_geocode_result[0]['formatted_address']
            return location_name
        else:
            return f"{latitude:.6f}, {longitude:.6f}"
            
    except Exception as e:
        print(f"Reverse geocoding error: {e}")
        return f"{latitude:.6f}, {longitude:.6f}"  # Fallback on error
    
def create_final_selection_message(place, user_lat, user_lon):
    """Create detailed selection message"""
    distance = calculate_distance(user_lat, user_lon, place['location']['lat'], place['location']['lng'])
    
    # Price level emojis
    price_emojis = {0: '💰', 1: '💵', 2: '💵💵', 3: '💵💵💵', 4: '💵💵💵💵'}
    price_display = price_emojis.get(place.get('price_level', 0), '💰')

    return f"🎊 <b>FINAL SELECTION!</b> 🎊\n\n" \
           f"🍽️ <b>{place['name']}</b>\n" \
           f"⭐ {place['rating']}/5 | {price_display}\n" \
           f"📍 {distance:.0f}m away\n" \
           f"🏠 {place.get('vicinity', 'No address')}\n\n" \
           f"_{place.get('description', 'No description available')}_\n\n" \
           f"🗺️ <a href=\"https://www.google.com/maps/search/?api=1&query={place.get('name', 'Unknown')}&query_place_id={place.get('place_id', '')}\">Open in Maps</a>"