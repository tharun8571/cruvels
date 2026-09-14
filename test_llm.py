import os
from dotenv import load_dotenv
from src.llm.model import get_llm
from src.config import get_settings

load_dotenv()

def main():
    settings = get_settings()
    provider = settings["llm"]["provider"]
    model_name = settings["llm"]["model"]
    
    print(f"LLM Provider configured: {provider}")
    print(f"LLM Model configured: {model_name}")
    
    try:
        llm = get_llm()
        print("LLM Instance initialized successfully:")
        print(llm)
        
        # Perform test invocation if API key is provided
        api_key = os.getenv("MOONSHOT_API_KEY")
        if api_key and api_key != "your-moonshot-key-here":
            print("\nSending test prompt to Kimi model...")
            response = llm.invoke("Hello, introduce yourself briefly.")
            print("Response from Kimi model:")
            print(response.content)
        else:
            print("\n[NOTE] MOONSHOT_API_KEY is not set or using placeholder key in .env.")
            print("To invoke the Kimi LLM API live, set your actual key in .env:")
            print("  MOONSHOT_API_KEY=your-actual-api-key")

    except Exception as e:
        print("Error initializing LLM:", e)

if __name__ == "__main__":
    main()
