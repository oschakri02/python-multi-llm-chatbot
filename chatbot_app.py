#Importing Required Packages..........

import os
import time
import json
import gradio as gr

from groq import Groq
from google import genai
from google.genai import types
from tools_integration import get_weather
from dotenv import load_dotenv

#Loading API Keys............
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from the environment.")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from the environment.")

# Initialize API clients............
gemini_client = genai.Client(api_key = GEMINI_API_KEY)
groq_client = Groq(api_key = GROQ_API_KEY)

#Personalities Dictionary(helps models to generate output based on personalities)............
personalities = {
    "Friendly" : """You are a friendly, enthusiastic, and highly encouraging study assistant. 
                    Your goal is to break down complex concepts into simple, beginner-friendly explanations. 
                    Use analogies and real-world examples that beginners can relate to. Always ask a follow-up question to check understanding.""",
    "Professional" : """You are a professional, respectful, and reliable study assistant. 
                        Your goal is to provide clear, accurate, and well-structured explanations that help users understand concepts efficiently. 
                        Use appropriate terminology, practical examples, and organized steps to communicate information effectively. 
                        Always maintain a respectful tone and ask a relevant follow-up question to clarify the user's needs or check understanding.""",
    "Creative" : """You are a creative, imaginative, and engaging study assistant. Your goal is to make complex concepts easier to understand by presenting them in interesting and memorable ways. 
                    Use storytelling, analogies, thought experiments, visual descriptions, and real-world examples that beginners can relate to. 
                    Encourage curiosity and explore different ways of understanding a concept. 
                    Always ask a thought-provoking follow-up question to encourage further learning.""",
    "Humorous" : """You are a humorous, playful, and approachable study assistant. Your goal is to make learning enjoyable by explaining complex concepts in simple, beginner-friendly ways. 
                    Use lighthearted humor, clever analogies, relatable real-world examples, and occasional jokes to make explanations memorable without sacrificing accuracy. 
                    Adapt your humor to the topic and the user's preferences, and avoid jokes when they would be inappropriate. 
                    Always ask a friendly follow-up question to check understanding or encourage the user to continue learning.""",
    "Critical Thinker" : """You are a critical-thinking, logical, and open-minded study assistant. Your goal is to help users understand concepts deeply by examining assumptions, evaluating evidence, identifying gaps in reasoning, and exploring alternative perspectives. 
                            Break complex concepts into clear, beginner-friendly explanations and use logical reasoning, analogies, and real-world examples to support understanding. 
                            Distinguish facts from assumptions and opinions, acknowledge uncertainty when appropriate, and respectfully challenge incorrect or unsupported conclusions. 
                            Rather than simply giving answers, encourage users to think independently by asking insightful follow-up questions that test their reasoning and deepen their understanding."""
}

# Converting previous Gradio messages into plain text

def get_text(content):
    if isinstance(content, str):
        return content

    # Handle text blocks if Gradio provides content as a list
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return ""

#Gemini API to generate response based on given input....................

def gemini_chatbot(question, history, persona):
    llm_messages = []

    #Converting earlier conversation into Gemini's format

    for item in history:
        role = item.get("role")
        content = get_text(item.get("content"))
        if not content:
            continue
        if role == "user":
            llm_messages.append(
                {
                    "role" : "user",
                    "parts" : [{"text" : content}]
                }
            )
        elif role == "assistant":
            llm_messages.append(
                {
                    "role" : "model",
                    "parts" : [{"text" : content}]
                }
            )

    # Adding the latest question to generate response..............

    llm_messages.append(
        {
            "role" : "user",
            "parts" : [{"text" : question}]
        }
    )

    gemini_response = gemini_client.models.generate_content(
        model = "gemini-3.8-flash",
        config = types.GenerateContentConfig(
            system_instruction = personalities[persona],
            temperature = 0.4,
            max_output_tokens = 1000
        ),
        contents = llm_messages        
    )
    return gemini_response.text or "Gemini Returned an empty response."

#Getting Real Time data using tool calling concept in LLM's....................
tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name like Mumbai, London"
            }
            },
      "required": ["location"]
           }
       }
   }
]

#Groq API to generate respone based on given input....................

def groq_chatbot(question, history, persona):

    llm_messages = [
        {
            "role": "system",
            "content": personalities[persona]
        }
    ]

    # Convert previous conversation into Groq's format......................
    for item in history:
        role = item.get("role")
        content = get_text(item.get("content"))

        if role in ("user", "assistant") and content:
            llm_messages.append(
                {
                    "role": role,
                    "content": content
                }
            )

    # Add the latest user question....................................
    llm_messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # First API request: ask Groq whether a tool is needed or not...................
    groq_response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=llm_messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.4,
        max_completion_tokens=1000
    )

    assistant_message = groq_response.choices[0].message

    # Check whether Groq requested a tool..............................
    if assistant_message.tool_calls:

        # Process the first tool call.......................
        tool_call = assistant_message.tool_calls[0]

        arguments = json.loads(tool_call.function.arguments)
        location = arguments["location"]

        # Execute the weather function......................
        weather_data = get_weather(location)

        # Add the assistant's tool-call message.......................
        llm_messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        # Add the weather result..................................
        llm_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "get_weather",
                "content": json.dumps(weather_data)
            }
        )

        # Second API request: generate the final response.......................
        final_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=llm_messages,
            max_completion_tokens=1000
        )

        return (
            final_response.choices[0].message.content
            or "Groq returned an empty response."
        )

    # Return a normal response if no tool was requested............................
    return (
        assistant_message.content
        or "Groq returned an empty response."
    )
    

#Main AI Models Control room..........................

def models(message, history, provider, persona):
    if not message or not message.strip():
        return "Please enter a question."

    start_time = time.perf_counter()

    try:
        if provider == "Gemini":
            answer =  gemini_chatbot(message, history, persona)
        elif provider == "Groq":
            answer = groq_chatbot(message, history, persona)
        else:
            return "Please select a valid AI provider."
        elapsed = time.perf_counter() - start_time
        print(f"{provider} response time: {elapsed:.2f} seconds")
        return answer
    
    except Exception as error:
        print(f"{provider} error: {type(error).__name__}: {error}")
        return (
            f"{provider} request failed ({type(error).__name__}).\n"
            "Check the VS Code Terminal for Error Details."
        )


#Gradio Chat InterFace

with gr.Blocks() as demo_bot:

    gr.Markdown("# 🤖 My AI Chatbot")
    gr.Markdown(
        "Choose an AI provider and personality, "
        "then start your conversation."
    )

    with gr.Row():
        provider = gr.Dropdown(
            choices=["Gemini", "Groq"],
            value="Gemini",
            label="Select AI Model"
        )

        persona = gr.Dropdown(
            choices=list(personalities.keys()),
            value="Friendly",
            label="Select Personality"
        )

    gr.ChatInterface(
        fn=models,
        additional_inputs=[provider, persona],
        title="Chat with AI",
        description="Your conversation history is passed to the selected AI model.",
        save_history=True
    )


demo_bot.launch(debug = True)  