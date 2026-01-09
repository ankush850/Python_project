import requests
import json  # For pretty-printing JSON if needed
from datetime import datetime  # For handling timestamps

def get_weather(api_key, city, units='metric'):
    """
    Fetches current weather data for a given city using OpenWeatherMap API.
    
    Args:
        api_key (str): Your OpenWeatherMap API key.
        city (str): Name of the city (e.g., 'London' or 'London,UK').
        units (str): Units for temperature ('metric' for Celsius, 'imperial' for Fahrenheit, 'standard' for Kelvin).
    
    Returns:
        dict: Parsed weather data if successful, None otherwise.
    """
    try:
        # Corrected base URL for city-based queries
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        
        # Parameters for the API request
        params = {
            'q': city,
            'appid': api_key,
            'units': units
        }
        
        # Make the GET request with a timeout for robustness
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse JSON response
        data = response.json()
        
        # Check if the response is successful
        if data.get('cod') == 200:
            # Extract relevant data
            weather_description = data['weather'][0]['description'].capitalize()
            temperature = data['main']['temp']
            feels_like = data['main']['feels_like']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            country = data['sys']['country']
            sunrise = datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S')
            sunset = datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S')
            
            # Print formatted output
            print(f"\nWeather in {city}, {country}:")
            print(f"Description: {weather_description}")
            print(f"Temperature: {temperature}°{'C' if units == 'metric' else 'F' if units == 'imperial' else 'K'}")
            print(f"Feels like: {feels_like}°{'C' if units == 'metric' else 'F' if units == 'imperial' else 'K'}")
            print(f"Humidity: {humidity}%")
            print(f"Wind Speed: {wind_speed} m/s")
            print(f"Sunrise: {sunrise}")
            print(f"Sunset: {sunset}")
            
            # Return the full data for further use
            return data
        else:
            print(f"API Error: {data.get('message', 'Unknown error')}")
            return None
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
    except KeyError as key_err:
        print(f"Data parsing error: Missing key {key_err}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return None

def get_forecast(api_key, city, units='metric'):
    """
    Fetches 5-day weather forecast for a given city.
    
    Args:
        api_key (str): Your OpenWeatherMap API key.
        city (str): Name of the city.
        units (str): Units for temperature.
    """
    try:
        base_url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            'q': city,
            'appid': api_key,
            'units': units
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') == '200':
            print(f"\n5-Day Forecast for {city}:")
            for item in data['list'][:5]:  # Show first 5 entries (every 3 hours)
                dt = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d %H:%M:%S')
                temp = item['main']['temp']
                desc = item['weather'][0]['description'].capitalize()
                print(f"{dt}: {temp}°{'C' if units == 'metric' else 'F'}, {desc}")
        else:
            print(f"API Error: {data.get('message', 'Unknown error')}")
    
    except Exception as e:
        print(f"Error fetching forecast: {e}")

if __name__ == "__main__":
    # Note: Replace with your actual API key. This one is a placeholder.
    api_key = "afe72a4da5b0392c920e53f98a0b7907"
    
    city = input("Enter the city name (e.g., 'London' or 'London,UK'): ").strip()
    units = input("Enter units ('metric' for Celsius, 'imperial' for Fahrenheit, 'standard' for Kelvin) [default: metric]: ").strip() or 'metric'
    
    # Get current weather
    weather_data = get_weather(api_key, city, units)
    
    # Optionally get forecast if current weather was successful
    if weather_data:
        get_forecast_choice = input("\nGet 5-day forecast? (y/n): ").strip().lower()
        if get_forecast_choice == 'y':
            get_forecast(api_key, city, units)
    
    # Example of using the full library: Save response to a file
    if weather_data:
        save_choice = input("\nSave weather data to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            with open('weather_data.json', 'w') as f:
                json.dump(weather_data, f, indent=4)
            print("Data saved to weather_data.json")
