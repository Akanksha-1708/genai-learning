from dotenv import load_dotenv
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    Use this tool when the user asks for arithmetic calculations.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Unable to calculate the expression."
    
@tool
def currency_converter(
    amount: float,
    from_currency: str,
    to_currency: str
) -> str:
    """
    Convert an amount from one currency to another.
    Use this tool when the user asks for currency conversion.
    """
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "INR": 88.0
    }
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    if from_currency not in rates or to_currency not in rates:
        return "Currency not supported."

    usd_amount = amount / rates[from_currency]
    converted_amount = usd_amount * rates[to_currency]
    return (
        f"{amount} {from_currency} = "
        f"{converted_amount:.2f} {to_currency}"
    )

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.
    This is a sample weather tool for demonstrating tool calling.
    """
    weather_data = {
        "kanpur": "Sunny, 32°C",
        "delhi": "Cloudy, 30°C",
        "mumbai": "Humid, 29°C",
        "bangalore": "Partly cloudy, 25°C"
    }
    return weather_data.get(
        city.lower(),
        f"Weather information for {city} is not available."
    )

tools = [
    calculator,
    currency_converter,
    get_weather
]

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.2,
    max_new_tokens=500
)

model = ChatHuggingFace(llm=llm)
model_with_tools = model.bind_tools(tools)

tool_map = {
    tool.name: tool
    for tool in tools
}

def run_assistant(question):
    messages = [
        HumanMessage(content=question)
    ]

    response=model_with_tools.invoke(messages)
    print("\n[Tool Calls]")
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(
                f"-{tool_call['name']}"
                f"({tool_call['args']})"
            )
    else:
        print("-None")
    messages.append(response)
    
    if not response.tool_calls:
        return response.content
    
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        selected_tool = tool_map.get(tool_name)
        if selected_tool is None:
            continue
        tool_result = selected_tool.invoke(tool_args)
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )

    final_response = model_with_tools.invoke(messages)
    return final_response.content

def main():
    print("=" * 70)
    print("TOOL-CALLING AI ASSISTANT")
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

        print("\nAssistant:")
        try:
            answer = run_assistant(question)
            print(answer)
        except Exception as error:
            print(f"Error: {error}")

if __name__ == "__main__":
    main()