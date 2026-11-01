#Importing JSON and Requests module to communicate with real time weather data.................

import os
import json
from urllib import response
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("WEATHER_API_KEY")

#Creating Weather Function.........

def get_weather(location: str) -> dict:
    """
    Fetch current weather information for a city
    using the OpenWeatherMap API.
    """

    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        return {
            "error": "WEATHER_API_KEY is missing from the environment."
        }

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": location,
        "units": "metric",
        "appid": api_key
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        if data.get("cod") != 200:
            return {
                "error": data.get(
                    "message",
                    "Unable to retrieve weather information."
                )
            }

        return {
            "location": data["name"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"]
        }

    except requests.exceptions.RequestException as error:
        return {
            "error": f"Weather API request failed: {error}"
        }

    except (ValueError, KeyError, IndexError) as error:
        return {
            "error": f"Unable to process weather data: {error}"
        }