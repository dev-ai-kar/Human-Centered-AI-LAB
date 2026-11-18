# Lab 9 - Multi-Agent Systems: the "What to Wear" crew.
# Same weather function as Lab 5. The work is split across a team of agents
# (weather, stylist, activity planner) coordinated by a manager, built with the
# OpenAI Agents SDK. requirements.txt needs: openai-agents (imported as `agents`).

import asyncio
import os
import time

import requests
import streamlit as st
from agents import Agent, ModelSettings, Runner, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


# ---------------------------------------------------------------------------
# Part B: the Lab 5 weather function, copied exactly, plus a tool wrapper.
# ---------------------------------------------------------------------------
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


@function_tool
def get_weather(location: str = "Syracuse, NY") -> dict:
    """Get the current weather and today's forecast for one location.

    Accepts a city ("Syracuse, NY", "Lima, Peru"), a ZIP code, a 3-letter
    airport code, or a landmark. Uses Syracuse, NY when no location is given.
    Looks up one place per call.

    Returns the current temperature and feels-like temperature (F), conditions,
    humidity, wind speed (mph), precipitation (mm), visibility (miles), today's
    highest chance of rain, high and low, sunrise and sunset, and today's
    forecast in 3-hour steps. If the place cannot be found, the call fails with
    an error message.
    """
    return get_current_weather(location)  # the Lab 5 function


# ---------------------------------------------------------------------------
# Part C: the team
# ---------------------------------------------------------------------------
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY to .streamlit/secrets.toml")
    st.stop()

MODEL = "gpt-5-mini"

weather_agent = Agent(
    name="Weather agent",
    instructions=(
        "You are a weather reporter. Call get_weather for the location you are "
        "given, then summarize the facts in a few short lines: current temperature "
        "and feels-like, conditions, today's high and low, rain chance, wind, and "
        "how conditions change through the day. Report facts only; do not give "
        "clothing or activity advice. If the location cannot be found, say so."
    ),
    tools=[get_weather],
    model=MODEL,
)

stylist_agent = Agent(
    name="Stylist agent",
    instructions=(
        "You are a practical stylist. You receive a weather summary. Recommend what "
        "to wear today: layers, outerwear, footwear, and accessories (umbrella, hat, "
        "sunglasses, gloves). Tie each choice to a weather fact. Keep it to a short "
        "bulleted list. Do not suggest activities."
    ),
    model=MODEL,
)

activity_agent = Agent(
    name="Activity agent",
    instructions=(
        "You are a local activity planner. You receive a location and a weather "
        "summary. Suggest 3 activities that fit the conditions, mixing outdoor and "
        "indoor options, and say the best time of day for outdoor ones. Keep it to a "
        "short bulleted list. Do not give clothing advice."
    ),
    model=MODEL,
)

manager_agent = Agent(
    name="Manager",
    instructions=(
        "You coordinate a team that tells the user what to wear and do today. "
        "Always call weather_report first with the user's location. Then pass that "
        "weather summary (and the location) to style_advice and to activity_ideas. "
        "Combine their answers into one friendly reply with a one-line weather "
        "overview, a 'What to wear' section, and a 'What to do' section. Do not add "
        "advice of your own. If the weather report says the location could not be "
        "found, skip the other tools and ask the user for a real place."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="weather_report",
            tool_description="Looks up today's weather for a location and returns a factual summary. Input: the location.",
        ),
        stylist_agent.as_tool(
            tool_name="style_advice",
            tool_description="Recommends an outfit for today. Input: the weather summary.",
        ),
        activity_agent.as_tool(
            tool_name="activity_ideas",
            tool_description="Suggests activities that fit today's weather. Input: the location and weather summary.",
        ),
    ],
    model=MODEL,
)


# ---------------------------------------------------------------------------
# Part E: one agent with the same tool and one combined prompt (Lab 6 prompt).
# ---------------------------------------------------------------------------
single_agent = Agent(
    name="Single agent",
    instructions=(
        "You are a helpful style and activity advisor for daily weather planning. "
        "Use the weather tool to check conditions, then recommend appropriate "
        "clothes for today and suggest outdoor activities that fit the conditions."
    ),
    tools=[get_weather],
    model=MODEL,
)


# ---------------------------------------------------------------------------
# Bonus: triage with handoffs
# ---------------------------------------------------------------------------
packing_agent = Agent(
    name="Packing agent",
    handoff_description="Builds a packing list for a trip to another city.",
    instructions=prompt_with_handoff_instructions(
        "You have taken over from the triage agent and answer the user yourself. "
        "Always call get_weather for the destination first, then write a packing list."
    ),
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="required"),  # call a tool first
    model=MODEL,
)

outfit_team = manager_agent.clone(
    name="Outfit team",
    handoff_description="What to wear and do today.",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Hand off trips to the Packing agent and today's outfits to the Outfit team.",
    handoffs=[packing_agent, outfit_team],
    model=MODEL,
)


# ---------------------------------------------------------------------------
# Runner helper: one event loop for the whole app (avoids "Event loop is closed").
# ---------------------------------------------------------------------------
@st.cache_resource
def get_event_loop():
    return asyncio.new_event_loop()


def run(agent, prompt):
    start = time.perf_counter()
    result = get_event_loop().run_until_complete(Runner.run(agent, prompt))
    return result, time.perf_counter() - start


def show_behind_the_scenes(result, seconds):
    usage = result.context_wrapper.usage
    with st.expander("Behind the scenes"):
        st.write(
            f"Model requests: {usage.requests} · Tokens: {usage.total_tokens:,} · "
            f"Time: {seconds:.1f}s · Answered by: {result.last_agent.name}"
        )
        for item in result.new_items:
            if item.type == "tool_call_item":
                st.write(f"{item.agent.name} called {item.raw_item.name}: {item.raw_item.arguments}")
            elif item.type == "tool_call_output_item":
                st.write(f"result: {str(item.output)[:300]}")
            elif item.type == "handoff_output_item":
                st.write(f"handoff: {item.source_agent.name} -> {item.target_agent.name}")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Lab 9: The What to Wear Crew")
st.write("A team of agents checks the weather, picks an outfit, and plans your day.")

mode = st.radio(
    "Who should answer?",
    ["Team (manager + 3 specialists)", "Single agent", "Triage (bonus)"],
    horizontal=True,
)

if mode == "Triage (bonus)":
    prompt = st.text_input(
        "Ask anything",
        value="I'm flying to Lima, Peru tomorrow for 3 days. What should I pack?",
    )
    agent = triage_agent
else:
    location = st.text_input("City or location", value="Syracuse, NY")
    prompt = f"What should I wear and do today in {location}?"
    agent = manager_agent if mode.startswith("Team") else single_agent

if st.button("Get advice"):
    try:
        with st.spinner("The agents are working... (a team run can take 20-40 seconds)"):
            result, seconds = run(agent, prompt)
    except Exception as exc:
        st.error(str(exc))
    else:
        with st.chat_message("assistant"):
            st.write(result.final_output)
        show_behind_the_scenes(result, seconds)


# ---------------------------------------------------------------------------
# Part D answers (from a team run on "Syracuse, NY")
# ---------------------------------------------------------------------------
# a. The manager called weather_report first ({"input": "Syracuse, NY"}), then
#    style_advice and activity_ideas (both in the same turn, after the weather
#    came back). get_weather itself never shows up in the list, because it was
#    called inside the weather agent's own run, not by the manager.
# b. A summary, not the raw forecast dict. The stylist got a few lines of text
#    ("Current: 46°F (feels like 42°F), Partly Cloudy. Today's high 55°F ...").
#    The weather agent wrote that summary, and the manager (an LLM) then copied
#    and slightly reworded it into the style_advice input. The stylist never
#    saw the tool's JSON, so any detail the summary left out (humidity,
#    visibility, sunset) was lost to it.
# c. 7 model requests but only 3 visible tool calls. Each agent-as-tool is a
#    separate run with its own model calls: the weather agent used 2 (one to
#    call get_weather, one to write the summary), the stylist 1 and the activity
#    agent 1. The manager used 3: one to call weather_report, one to call the
#    other two tools, and one to write the combined final answer.
#    2 + 1 + 1 + 3 = 7.
#
# Part E (gpt-5-mini, one run each):
#                     Syracuse, NY          Lima, Peru
#   Single agent      2 req, 1,689 tok, 12s  2 req, 1,633 tok, 8s
#   Team              7 req, 7,610 tok, 33s  7 req, 6,804 tok, 33s
# The team costs ~4x the tokens and ~3-4x the time. Both answers were good; the
# team's is more structured and ties every clothing item to a weather number,
# but the single agent's was just as useful (it even added °C for Lima).
