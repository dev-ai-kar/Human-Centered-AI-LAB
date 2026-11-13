# Lab 6 - LangChain: the "What to Wear" bot, rebuilt.
# Same weather function as Lab 5. LangChain's agent replaces the JSON tool schema,
# the two hand-written API calls, and the message bookkeeping between them.
# requirements.txt needs: langchain, langchain-openai

import os

import requests
import streamlit as st
from langchain.agents import create_agent


# ---------------------------------------------------------------------------
# Part B: the Lab 5 weather function, used directly as the tool.
# No JSON schema and no json import: LangChain builds the schema from the
# function name, type hints, and docstring. The docstring is sent to the model
# as the tool description, so it is part of the prompt.
# ---------------------------------------------------------------------------
def get_weather_data(location: str = "Syracuse, NY") -> dict:
    """Get the current weather and today's forecast for one location.

    Accepts a city ("Syracuse, NY", "Lima, Peru"), a ZIP code, a 3-letter
    airport code, or a landmark ("Eiffel Tower"). Uses Syracuse, NY when no
    location is given. Looks up one place per call, so call it once for each
    place the user mentions.

    Returns the current temperature and feels-like temperature (F), conditions,
    humidity, wind speed (mph), precipitation (mm), visibility (miles), today's
    highest chance of rain, high and low, sunrise and sunset, and today's
    forecast in 3-hour steps from midnight to 3 PM. If the lookup fails, it
    returns {"error": ...} instead; try again with a nearby city name.
    """
    if not location or not location.strip():
        location = "Syracuse, NY"

    url = f"https://wttr.in/{location}?format=j1"

    # Changed from Lab 5: lookup failures are returned instead of raised, so the
    # agent sees what went wrong and can try again (for example, a nearby city).
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as exc:
        return {"error": f"Weather service unreachable: {exc}"}

    if response.status_code != 200:
        return {"error": f"wttr.in error: status {response.status_code}"}

    try:
        data = response.json()
        current = data["current_condition"][0]
    except (ValueError, KeyError, IndexError):
        return {"error": f"Could not find a location named {location}"}

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


# ---------------------------------------------------------------------------
# Part C: build the agent
# ---------------------------------------------------------------------------
# Lab 5 passed the key to OpenAI(api_key=...). LangChain's OpenAI integration
# reads it from the OPENAI_API_KEY environment variable instead.
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY to .streamlit/secrets.toml")
    st.stop()

agent = create_agent(
    # Same model as Lab 5, so the comparison is direct.
    # Bonus: add langchain-anthropic to requirements.txt, copy the key with
    # os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"],
    # then change only this line to model="claude-sonnet-4-6".
    model="openai:gpt-4o-mini",
    tools=[get_weather_data],
    # Lab 5 system prompt, plus the instruction from Lab 5's follow-up prompt
    # (there is no hand-written second call anymore, so it lives here now).
    system_prompt=(
        "You are a helpful style and activity advisor for daily weather planning. "
        "Use the weather tool to check conditions, then recommend appropriate "
        "clothes for today and suggest outdoor activities that fit the conditions."
    ),
)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Lab 6: What to Wear Bot (LangChain)")
st.write("Enter a city to get outfit and activity suggestions based on the current weather.")

location = st.text_input("City or location", value="Syracuse, NY")

if st.button("Get clothing advice"):
    try:
        with st.spinner("Checking the weather and generating advice..."):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": f"What should I wear in {location}?"}]}
            )
    except Exception as exc:
        st.error(str(exc))
    else:
        # The agent's answer is the last message in the conversation.
        with st.chat_message("assistant"):
            st.write(result["messages"][-1].content)

        # Part D: every message the agent sent and received.
        with st.expander("Behind the scenes"):
            for message in result["messages"]:
                st.write(f"**{message.type}**")
                st.write(message.content)
                if message.type == "ai" and message.tool_calls:
                    st.write(message.tool_calls)
