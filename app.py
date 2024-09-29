from dotenv import load_dotenv
import chainlit as cl
from movie_functions import get_now_playing_movies, get_showtimes, buy_ticket
import json

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI

 
# client = AsyncOpenAI()
client = AsyncOpenAI(base_url="http://0.0.0.0:4000")

#available models:
# remote: gpt-4o, gpt4o-mini, gemini-flash, gemini-1.5-pro
# ollama: gemma2, qwen2.5, llama3.1

gen_kwargs = {
    "model": "mistral",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a helpful assistant whose job is to chat about movies with the user in a generally cinephilic manner.

If the response does not require a function call, respond as a normal chatbot.

### Available Functions
When calling functions, do not return a code block and do not respond with any commentary. Return a pure json object ready to be parsed.

1. **get_now_playing_movies() -> list**
   - **Purpose:** Retrieves a list of movies that are currently playing in theaters.
   - **Parameters:** None.
   - **Returns:** A list of movie titles currently showing.

2. **get_showtimes(title: str, location: str) -> list**
   - **Purpose:** Retrieves showtimes for a specific movie at a given location.
   - **Parameters:**
     - `title`: The title of the movie.
     - `location`: The city or region where the user wants to find showtimes.
   - **Returns:** A list of showtime strings (e.g., ["5:00 PM", "8:00 PM"]).

3. **buy_ticket(theater: str, movie: str, showtime: str) -> str**
   - **Purpose:** Purchases a ticket for a movie at a specific theater and showtime.
   - **Parameters:**
     - `theater`: The name of the theater.
     - `movie`: The title of the movie.
     - `showtime`: The showtime of the movie.
   - **Returns:** A confirmation message indicating the ticket purchase.

4. **confirm_ticket_purchase(theater: str, movie: str, showtime: str) -> str**
   - **Purpose:** Confirms the ticket purchase for a movie at a specific theater and showtime.
   - **Parameters:**
     - `theater`: The name of the theater.
     - `movie`: The title of the movie.
     - `showtime`: The showtime of the movie.
   - **Returns:** A confirmation message indicating the ticket purchase.


### Usage Guidelines

- **Current Movies:**
  - **Trigger:** User asks about movies currently playing.
  - **Action:** Use `get_now_playing_movies` to retrieve the list. If the user requested it, tell them what is playing.
  
- **Showtimes:**
  - **Trigger:** User inquires about showtimes for a specific movie.
  - **Action:** Ensure both `title` and `location` are provided. Use `get_showtimes(title, location)`.
  - **Note:** If either `title` or `location` is missing, ask the user to provide the missing information.

- **Ticket Purchase:**
  - **Trigger:** User wants to purchase a ticket for a movie.
  - **Action:** Use `buy_ticket(theater, movie, showtime)` with the provided `theater`, `movie`, and `showtime`.
  - **Note:** Ensure all parameters are correctly provided.

- **Confirmation:**
  - **Trigger:** BEFORE a ticket purchase, the user might want to confirm the purchase.
  - **Action:** Use `confirm_ticket_purchase(theater, movie, showtime)` to confirm the ticket purchase before actually purchasing.
  - **Note:** This function should be used BEFORE a successful ticket purchase.

### Examples

- **User:** "What movies are playing right now?"
- **Assistant:**
    {
        "function_name": "get_now_playing_movies",
        "arguments": {},
        "rationale": "The user wants to know the list of movies currently playing."
    }

- **User:** "Show me the showtimes for Inception in Los Angeles."
- **Assistant:**
    {
        "function_name": "get_showtimes",
        "arguments": {
            "title": "Inception",
            "location": "Los Angeles"
        },
        "rationale": "The user is requesting showtimes for 'Inception' in 'Los Angeles'."
    }

- **User:** "Tell me more about the movie Inception."
- **Assistant:**
    {
        "function_name": "get_movie_details",
        "arguments": {
            "title": "Inception"
        },
        "rationale": "The user wants detailed information about 'Inception'."
    }

- **User:** "I love sci-fi movies!"
  - **Assistant:** "That's great! Sci-fi movies offer some of the most imaginative and thought-provoking stories."

### JSON Response Schema

When calling a function, respond with a JSON object following this schema. Do not return a code block. do not prefix with any other text:

{
    "function_name": "function_name",
    "arguments": {
        "arg1": "value1",
        "arg2": "value2",
        ...
    },
    "rationale": "Explain why you are calling the function"
}
"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    # Get message history from the session
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    response_message = await generate_response(client, message_history, gen_kwargs)

    
    while True:
        # Generate a response from the assistant
        function_called = False
        content = response_message.content

        # Check if the response is surrounded by ```json and ``` markers
        if content.startswith("```json") and content.endswith("```"):
            # Strip the code block markers
            print('code block detected')
            content = content[7:-3].strip()  # Removes the first 7 characters (```json) and last 3 characters (```)

        print('here is the content', content)
        # Check if the response is a function call (i.e., valid JSON object)
        if content.strip().startswith('{'):
            try:
                # Parse the JSON object
                function_call = json.loads(content.strip())
                print('function detected', function_call)
                
                # Check if it's a valid function call
                if "function_name" in function_call and "rationale" in function_call:
                    function_name = function_call["function_name"]
                    function_called = True  # Mark that a function was called
                    print('function name detected', function_name)
                    # message_history.append({"role": "system", "content": f"You just called: {function_call}"})
                
                    # Handle the function call
                    if function_name == "get_now_playing_movies":
                        movies = await get_now_playing_movies()
                        message_history.append({"role": "system", "content": f"These are the movies currently playing:\n\n{movies}"})
                        response_message = await generate_response(client, message_history, gen_kwargs)
                    elif 'get_showtimes' in function_name:
                        title = function_call["arguments"]["title"]
                        location = function_call["arguments"]["location"]
                        showtimes = await get_showtimes(title, location)
                        message_history.append({"role": "system", "content": f"Here are the showtimes for {title} in {location}:\n\n{showtimes}"})
                        response_message = await generate_response(client, message_history, gen_kwargs)
                    elif 'buy_ticket' in function_name:
                        theater = function_call["arguments"]["theater"]
                        movie = function_call["arguments"]["movie"]
                        showtime = function_call["arguments"]["showtime"]
                        message_history.append({"role": "system", "content": f"Ticket purchased for {movie} at {theater} for {showtime}."})
                        response_message = await generate_response(client, message_history, gen_kwargs)
                    elif 'confirm_ticket_purchase' in function_name:
                        theater = function_call["arguments"]["theater"]
                        movie = function_call["arguments"]["movie"]
                        showtime = function_call["arguments"]["showtime"]
                        message_history.append({"role": "system", "content": f"Ticket purchased for {movie} at {theater} for {showtime}."})
                        response_message = await generate_response(client, message_history, gen_kwargs)
                        
                else:
                    # If there's no valid function call in the response, break the loop
                    print('no function detected')
                    break
            
            except json.JSONDecodeError:
                # If the response is not valid JSON (meaning it's not a function call), treat it as normal content
                message_history.append({"role": "assistant", "content": response_message.content})
                print('json decode error')
                break  # Exit the loop, as the bot is no longer calling functions

        else:
            # If the response is not a function call, append it as a normal assistant message and exit the loop
            message_history.append({"role": "assistant", "content": response_message.content})
            print('no function detected, normal message')
            break  # No more functions to call, so we exit the loop

        # Store updated message history in the session
        cl.user_session.set("message_history", message_history)

        # Exit the loop if no function was called
        if not function_called:
            break

    # Final update of message history
    cl.user_session.set("message_history", message_history)


if __name__ == "__main__":
    cl.main()