from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(expression:str)->str:
    """
    Calculate a mathematical expression.
    Use this tool for arithmetic calculations.
    """
    try:
        result=eval(expression,{"__builtins__":{}},{})
        return str(result)
    except Exception:
        return "Unable to calculate the expression"

@tool
def currency_convertor(amount:float,from_currency:str,to_currency:str)->str:
    """
    Convert an amount from one supported currency to another"""
    rates={
        "USD":1.0,
        "EUR":0.92,
        "GBP":0.79,
        "INR":88.0
    }
    from_currency=from_currency.upper()
    to_currency=to_currency.upper()
    if from_currency not in rates:
        return f"{from_currency} is not supported"
    if to_currency not in rates:
        return f"{to_currency} is not supported"
    usd_amount=amount/rates[from_currency]
    converted_amount=usd_amount*rates[to_currency]
    return(f"{amount} {from_currency}={converted_amount:.2f} {to_currency}")

@tool
def get_weather(city:str)->str:
    """Get sample weather information for a city"""
    weather_data = {
        "kanpur": "Sunny, 32°C",
        "delhi": "Cloudy, 30°C",
        "mumbai": "Humid, 29°C",
        "bangalore": "Partly cloudy, 25°C"
    }
    return weather_data.get(city.lower(),f"Weather information for {city} is not available")

tools=[calculator,currency_convertor,get_weather]

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
You are a helpful AI assistant.
You have access to several tools that can help you
complete specific tasks.
Use a tool when it is necessary or useful for the task.
You may answer directly using your own knowledge when
no tool is required.
The available tools are specialized:
- Calculator: mathematical calculations
- Currency Converter: currency conversions
- Weather: weather information for supported cities
Do not use an unrelated tool just because a tool is available.
For multi-step tasks, use the result of one action
to decide what to do next.
Do not invent tool results.
"""
)

def main():

    print("=" * 70)
    print("MULTI-TOOL AI AGENT")
    print("=" * 70)
    print("\nAvailable tools:")
    print("- Calculator")
    print("- Currency Converter")
    print("- Weather")
    print("\nType 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() == "exit":
            print("\nGoodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue
        try:
            response=agent.invoke(
                {"messages":[{"role":"user","content":question}]}
            )
            print("\nAssistant:")
            print(response["messages"][-1].content)

        except Exception as error:
            print(f"\nError: {error}")

if __name__ == "__main__":
    main()