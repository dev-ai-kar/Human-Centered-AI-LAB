import json

import requests
import streamlit as st
from openai import OpenAI


def get_current_weather(location):
    """Fetch current weather and key forecast details useful for clothing and activity advice."""
    if not location or not location.strip():
        location = "Syracuse, NY"

    url = f"https://wttr.in/{location}?format=j1"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception(f"wttr.in error: status {response.status_code}")

    try:
        data = response.json()
    except ValueError as exc:
        raise Exception(f"Could not find a location named {location}") from exc

    current = data["current_condition"][0]
    forecast = data.get("weather", [{}])[0]
    hourly = forecast.get("hourly", [])
    calculated_rain_chance = max(
        (int(entry.get("chanceofrain", 0)) for entry in hourly),
        default=0,
    )

    location_name = location
    nearest_area = data.get("nearest_area", [])
    if nearest_area:
        area_name = nearest_area[0].get("areaName", [{}])[0].get("value")
        if area_name:
            location_name = area_name

    astronomy = forecast.get("astronomy", [{}])[0]
    hourly_summary = []
    for entry in hourly[:6]:
        hourly_summary.append(
            {
                "time": entry.get("time", "Unknown"),
                "temperature_f": float(entry.get("tempF", current.get("temp_F", 0))),
                "description": entry.get("weatherDesc", [{"value": "Unknown"}])[0].get("value", "Unknown"),
                "rain_chance_percent": int(entry.get("chanceofrain", 0)),
                "wind_speed_mph": int(entry.get("windspeedMiles", 0)),
            }
        )

    return {
        "location": location_name,
        "temperature_f": float(current.get("temp_F", 0)),
        "feels_like_f": float(current.get("FeelsLikeF", current.get("temp_F", 0))),
        "description": current["weatherDesc"][0]["value"],
        "humidity": int(current.get("humidity", 0)),
        "wind_speed_mph": int(current.get("windspeedMiles", 0)),
        "precipitation_mm": float(current.get("precipMM", 0)),
        "visibility_miles": float(current.get("visibilityMiles", 0)),
        "rain_chance_percent": calculated_rain_chance,
        "day_high_f": float(forecast.get("maxtempF", current.get("temp_F", 0))),
        "day_low_f": float(forecast.get("mintempF", current.get("temp_F", 0))),
        "sunrise": astronomy.get("sunrise", "Unknown"),
        "sunset": astronomy.get("sunset", "Unknown"),
        "hourly_forecast": hourly_summary,
    }


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather and short forecast for a city or place so the assistant can recommend clothing and activities.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "A city, ZIP code, airport code, or landmark. If not provided, use Syracuse, NY.",
                }
            },
            "required": ["location"],
        },
    },
}


def get_wear_recommendation(location_text):
    """Use OpenAI tool calling to fetch weather and then ask the LLM for outfit advice."""
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except KeyError:
        raise ValueError(
            "OpenAI API key not found. Add OPENAI_API_KEY to .streamlit/secrets.toml"
        )

    client = OpenAI(api_key=openai_api_key)
    user_location = (location_text or "").strip() or "Syracuse, NY"

    first_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Give me clothing and activity advice for {user_location}.",
            }
        ],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        temperature=0.7,
    )

    assistant_message = first_response.choices[0].message

    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        tool_name = tool_call.function.name
        if tool_name == "get_current_weather":
            tool_args = json.loads(tool_call.function.arguments)
            location = tool_args.get("location") or user_location
            weather = get_current_weather(location)

            follow_up_prompt = (
                "Use the weather information below to recommend appropriate clothes for today "
                "and suggest outdoor activities that fit the conditions.\n\n"
                f"Weather data: {json.dumps(weather, indent=2)}"
            )

            follow_up = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful style and activity advisor for daily weather planning.",
                    },
                    {"role": "user", "content": follow_up_prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            return follow_up.choices[0].message.content.strip()

    if assistant_message.content:
        return assistant_message.content.strip()

    return "I couldn't generate advice for that location right now."


st.title("Lab 5: What to Wear Bot")
st.write("Enter a city to get outfit and activity suggestions based on the current weather.")

location = st.text_input("City or location", value="Syracuse, NY")

if st.button("Get clothing advice"):
    try:
        with st.spinner("Checking the weather and generating advice..."):
            advice = get_wear_recommendation(location)

        st.subheader("What to wear")
        st.markdown(advice)
    except Exception as exc:
        st.error(str(exc))
