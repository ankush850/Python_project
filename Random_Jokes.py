try:
    import pyjokes
except ImportError:
    print("Error: pyjokes library is not installed. Install it with 'pip install pyjokes'.")
    exit(1)

def generate_jokes(language="en", category="neutral", num_jokes=5):
    supported_languages = ['en', 'de', 'es', 'fr', 'it', 'pt']
    if language not in supported_languages:
        print(f"Warning: Language '{language}' not recognized. Defaulting to 'en'.")
        language = "en"
    
    supported_categories = ['neutral', 'chuck', 'all']
    if category not in supported_categories:
        print(f"Warning: Category '{category}' not supported. Defaulting to 'neutral'.")
        category = "neutral"
    
    if not isinstance(num_jokes, int) or num_jokes <= 0:
        print("Warning: Invalid number of jokes. Defaulting to 5.")
        num_jokes = 5
    
    try:
        jokes = pyjokes.get_jokes(language=language, category=category, number=num_jokes)
        
        if len(jokes) < num_jokes:
            print(f"Note: Only {len(jokes)} jokes available in category '{category}'. Displaying all.")
            num_jokes = len(jokes)
        
        print(f"\nHere are {num_jokes} random jokes in '{language}' from category '{category}':\n")
        for i, joke in enumerate(jokes, start=1):
            print(f"{i}. {joke}")
            print()
    
    except Exception as e:
        print(f"Error fetching jokes: {e}. Please check your inputs or try again.")

if __name__ == "__main__":
    print("Welcome to the Flexible Joke Generator!")
    
    language = input("Enter language (e.g., 'en' for English, default: 'en'): ").strip() or "en"
    category = input("Enter category ('neutral', 'chuck', 'all'; default: 'neutral'): ").strip() or "neutral"
    try:
        num_jokes = int(input("Enter number of jokes (default: 5): ").strip() or 5)
    except ValueError:
        print("Invalid number. Using default of 5.")
        num_jokes = 5
    
    generate_jokes(language=language, category=category, num_jokes=num_jokes)
