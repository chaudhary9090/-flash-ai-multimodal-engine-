"""
Weather Service: Open-Meteo Geocoding & Forecast Client.
Single-purpose design ready for LangChain Tool conversion in Phase 4.
"""

import re
import json
import urllib.request
import urllib.parse
from typing import Optional
from app.core.logging import logger

TYPO_CORRECTIONS = {
    "fraidabad": "faridabad",
    "fardabad": "faridabad",
    "delhi": "delhi",
    "mumbai": "mumbai",
    "vadodra": "vadodara",
    "bengaluru": "bangalore",
}


def get_wmo_description(code: int) -> str:
    codes = {
        0: "Clear Sky (Sunny)", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
        45: "Foggy", 51: "Light Drizzle", 61: "Slight Rain", 63: "Moderate Rain",
        65: "Heavy Rain", 80: "Rain Showers", 95: "Thunderstorm"
    }
    return codes.get(code, "Partly Cloudy")


def fetch_global_weather(query: str) -> Optional[str]:
    """
    Geocodes location and fetches real-time weather from Open-Meteo API.
    
    Signature: (query: str) -> Optional[str]
    Single-purpose contract ready for Phase 4 @tool wrapping.
    """
    clean_query = re.sub(r"[^\w\s]", "", query.lower().strip())
    
    city_match = re.search(r"(?:weather|temperature|temp|climate)\s+(?:in|of|at|for)?\s*([a-zA-Z\s]+)", clean_query)
    if not city_match:
        city_match = re.search(r"([a-zA-Z\s]+)\s+(?:weather|temperature|temp)", clean_query)

    city_raw = city_match.group(1).strip() if city_match else clean_query.replace("weather", "").replace("in", "").strip()
    city_name = TYPO_CORRECTIONS.get(city_raw, city_raw)
    if not city_name:
        city_name = "Delhi"

    try:
        encoded_city = urllib.parse.quote(city_name)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
        
        req = urllib.request.Request(geo_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as response:
            geo_data = json.loads(response.read().decode())
            results = geo_data.get("results")
            
            if not results and len(city_name) >= 4:
                fuzzy_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city_name[:4])}&count=1&language=en&format=json"
                with urllib.request.urlopen(urllib.request.Request(fuzzy_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=4) as f_resp:
                    results = json.loads(f_resp.read().decode()).get("results")

            if not results:
                return f"[WEATHER] Location '{city_name.title()}' not found."

            first_result = results[0]
            lat = first_result.get("latitude")
            lon = first_result.get("longitude")
            resolved_city = first_result.get("name")
            country = first_result.get("country", "")
            admin1 = first_result.get("admin1", "")
            location_label = f"{resolved_city}, {admin1}, {country}".replace(", ,", ",").strip(", ")

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        with urllib.request.urlopen(urllib.request.Request(weather_url, headers={"User-Agent": "Mozilla/5.0"}), timeout=4) as w_resp:
            cw = json.loads(w_resp.read().decode()).get("current_weather", {})
            temp = cw.get("temperature")
            wind = cw.get("windspeed")
            code = cw.get("weathercode", 0)
            desc = get_wmo_description(code)

            return (
                f"[LIVE WEATHER REPORT]\n"
                f"Location: {location_label}\n"
                f"- Temperature: {temp} C\n"
                f"- Condition: {desc}\n"
                f"- Wind Speed: {wind} km/h\n"
                f"- Coordinates: ({lat:.2f}, {lon:.2f})\n"
                f"- Data Source: Global Open-Meteo Live API"
            )
    except Exception as e:
        logger.error(f"Weather tool error: {e}")
        return None
