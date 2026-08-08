import os
import httpx
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from tools.map_tool import build_travel_map
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    http_client=httpx.Client(verify=False),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

SYSTEM_PROMPT = """You are an expert travel planner. Using the provided flight, hotel, weather, country, and currency data, create a detailed day-by-day travel itinerary.

Structure your response EXACTLY like this:

## ✈️ Trip Overview
- Route, dates, travellers, class

## 💺 Recommended Flight
- Pick the best flight from the options and explain why

## 🏨 Recommended Hotel
- Pick the highest-rated hotel and explain why

## 📅 Day-by-Day Itinerary
### Day 1 - [Date] - [Weather emoji + condition]
- Morning: ...
- Afternoon: ...
- Evening: ...
- 💡 Tip: ...

(repeat for each day)

## 💰 Budget Breakdown
- Flights: estimated cost
- Hotel: estimated cost
- Daily expenses: estimate per person per day
- Total estimate

## 🌍 Travel Tips
- Visa info
- Currency tips
- Weather advice
- Local customs
- Emergency contacts (embassy, local police)

Keep the tone friendly, practical and helpful. Use the weather data to suggest what to pack and best times for outdoor activities."""


def itinerary_agent(state: dict) -> dict:
    """
    Uses LLM to synthesize all collected data into a day-by-day travel itinerary.
    Also builds the Folium map.
    """
    # collect all data summaries from state
    flight_summary   = state.get("flight_summary",   "No flight data available.")
    hotel_summary    = state.get("hotel_summary",    "No hotel data available.")
    weather_summary  = state.get("weather_summary",  "No weather data available.")
    country_summary  = state.get("country_summary",  "No country data available.")
    currency_summary = state.get("currency_summary", "No currency data available.")

    origin      = state.get("origin", "")
    destination = state.get("destination", "")
    depart_date = state.get("depart_date", "")
    return_date = state.get("return_date", "")
    adults      = state.get("adults", 1)
    preferences = state.get("preferences", "")
    budget_inr  = state.get("budget_inr", 0)

    user_context = f"""
TRIP DETAILS:
- From: {origin} → To: {destination}
- Dates: {depart_date} to {return_date}
- Travellers: {adults} adult(s)
- Budget: {'INR {:,.0f}'.format(budget_inr) if budget_inr else 'Not specified'}
- Preferences: {preferences or 'None'}

FLIGHT OPTIONS:
{flight_summary}

HOTEL OPTIONS:
{hotel_summary}

WEATHER FORECAST:
{weather_summary}

COUNTRY INFORMATION:
{country_summary}

CURRENCY:
{currency_summary}

Please create a complete travel itinerary based on all the above data.
"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_context),
    ])

    itinerary = response.content

    # build the interactive map
    hotel_results = state.get("hotel_results", {})
    hotels        = hotel_results.get("hotels", []) if isinstance(hotel_results, dict) else []

    map_html = build_travel_map(
        origin_iata      = state.get("origin_iata", origin),
        destination_iata = state.get("destination_iata", destination),
        hotels           = hotels,
        destination_city = destination,
    )

    return {
        "itinerary": itinerary,
        "map_html":  map_html,
    }
