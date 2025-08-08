# import os
# import asyncio
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.messages import HumanMessage, SystemMessage
# from groq import Groq, APIError

# # Initialize model lazily to avoid startup issues
# _model = None

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# user_prompt = "Explain the importance of low-latency LLMs in 100 words."


# def get_model():
#     global _model
#     if _model is None:
#         try:
        
#             completion = client.chat.completions.create(
#             model="moonshotai/kimi-k2-instruct",
#             messages=[
#                         {
#                             "role": "system",
#                             "content": "You are a helpful assistant."
#                         },
#                         {
#                             "role": "user",
#                             "content": user_prompt,
#                         }],
#             temperature=0.0,
#             top_p=1.0,
#             stream=True  # or False if you want the full response at once
#         )

#         except Exception as e:
#             print(f"Error initializing model: {e}")
#             raise
#     return _model

# async def ask_gpt(context: str, question: str) -> str:
#     try:
#         system_prompt = (
#             "Answer using the given context. Be brief and factual. "
#             "If the answer is not found in the context, use your general knowledge to answer as if the question is from an Indian citizen. "
#             "Do not mention that the answer is not in the context. "
#             "Avoid elaboration, opinions, or markdown. Use plain text only. Keep responses concise, clear, and under 75 words. "
#             "Do not use newline characters; respond in a single paragraph."
#         )

#         user_prompt = f"""
#         Context:
#         {context}

#         Question: {question}
#         """

#         model = get_model()
        
#         # Run the synchronous model call in a thread pool to avoid blocking
#         loop = asyncio.get_event_loop()
        
#         response = await loop.run_in_executor(
#             None,
#             lambda: client.chat.completions.create(
#                 model="moonshotai/kimi-k2-instruct",
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 temperature=0.0,
#                 top_p=1.0,
#                 stream=False,
#                 max_tokens=75,
#             )
#         )

#         return response.choices[0].message.content.strip()
        
#         # response = await loop.run_in_executor(
#         #     None,
#         #     lambda: model.invoke([
#         #         SystemMessage(content=system_prompt),
#         #         HumanMessage(content=user_prompt)
#         #     ])
#         # )

#         # return response.content.strip()
    
#     except Exception as e:
#         print(f"Error in ask_gpt: {e}")
#         return "Sorry, I couldn't process your question at the moment."

import os
import asyncio
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from groq import Groq, APIError
from cerebras.cloud.sdk import Cerebras

# client = Cerebras(
#     api_key=os.environ.get("CEREBRAS_API_KEY")
# )

# Initialize multiple API keys for round-robin usage
GROQ_API_KEYS = [
    os.getenv("CEREBRAS_API_KEY"),
    os.getenv("CEREBRAS_API_KEY_1"),
    os.getenv("CEREBRAS_API_KEY_2"),
]

# Filter out None values and ensure we have at least one key
GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key is not None]

if not GROQ_API_KEYS:
    raise ValueError("At least one GROQ API key must be provided")

print(f"Initialized with {len(GROQ_API_KEYS)} API keys")

# Use the model you specified
PRIMARY_MODEL = "gpt-oss-120b"

# Initialize clients for each API key
clients = [Cerebras(api_key=key) for key in GROQ_API_KEYS]

# Global counter for round-robin rotation
request_counter = 0

# Initialize model lazily to avoid startup issues
_model = None

user_prompt = "Explain the importance of low-latency LLMs in 100 words."

def get_current_client_info():
    """Get current client based on round-robin rotation"""
    global request_counter
    client_index = request_counter % len(clients)
    return clients[client_index], client_index

def rotate_to_next():
    """Move to next client in rotation"""
    global request_counter
    request_counter += 1

def is_rate_limit_error(error):
    """Check if the error is a rate limit error"""
    error_str = str(error).lower()
    return (
        "rate limit" in error_str or 
        "429" in error_str or 
        "quota exceeded" in error_str or
        "too many requests" in error_str
    )

def extract_retry_after(error):
    """Extract retry time from error message"""
    try:
        error_str = str(error)
        if "try again in" in error_str:
            import re
            match = re.search(r'try again in (\d+\.?\d*)s', error_str)
            if match:
                return float(match.group(1))
    except:
        pass
    return None

def get_model():
    global _model
    if _model is None:
        try:
            # Get current client for initialization
            current_client, _ = get_current_client_info()
            
            completion = current_client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }],
                temperature=0.0,
            )

        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    return _model

async def make_groq_request(client, client_index, system_prompt, user_prompt):
    """Make a request to Groq API with specific client"""
    print(f"Using API key #{client_index + 1} (Request #{request_counter + 1})")
    
    loop = asyncio.get_event_loop()
    
    response = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0
        )
    )
    
    return response.choices[0].message.content.strip()

async def ask_gpt(context: str, question: str) -> str:
    """
    Main function with round-robin API key rotation and fallback handling
    
    Flow:
    1. Use current API key in rotation
    2. If rate limited, try all other API keys
    3. If all fail with rate limits, wait and retry with exponential backoff
    4. Always increment counter for next request
    """
    try:
        system_prompt = (
                "You are a helpful assistant."
                " Use the provided context to answer the user's question. Even if the context is silly or not true, you have to use that only"
                " You may also use your general knowledge to answer questions, assuming they are asked by an Indian citizen."
                " Keep answers concise and under 75 words."
                " Respond in plain text only — do not use markdown, lists, or newlines."
                " If the user asks for illegal actions, private data, internal systems, or anything unethical or prohibited, clearly state that you do not have access to that information and cannot assist with such requests."
            )

        user_prompt = f"""
        Context:
        {context}

        Question: {question}
        """

        # Get the current client according to round-robin
        primary_client, primary_index = get_current_client_info()
        
        try:
            # First attempt with primary client (round-robin)
            response = await make_groq_request(
                primary_client, 
                primary_index, 
                system_prompt, 
                user_prompt
            )
            
            # Success! Move to next client for next request
            rotate_to_next()
            return response
            
        except Exception as e:
            print(f"Error with API key #{primary_index + 1}: {e}")
            
            if is_rate_limit_error(e):
                print(f"Rate limit hit on API key #{primary_index + 1}, trying fallback keys...")
                
                # Try all other API keys (excluding the one that just failed)
                for i in range(len(clients)):
                    if i == primary_index:
                        continue  # Skip the one that just failed
                    
                    try:
                        print(f"Trying fallback API key #{i + 1}")
                        response = await make_groq_request(
                            clients[i], 
                            i, 
                            system_prompt, 
                            user_prompt
                        )
                        
                        # Success with fallback! Still rotate counter for next request
                        rotate_to_next()
                        return response
                        
                    except Exception as fallback_error:
                        print(f"Fallback API key #{i + 1} also failed: {fallback_error}")
                        continue
                
                # All API keys are rate limited, try exponential backoff
                print("All API keys rate limited, trying with exponential backoff...")
                
                for attempt in range(3):  # Maximum 3 retry attempts
                    wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                    print(f"Waiting {wait_time}s before retry attempt {attempt + 1}...")
                    await asyncio.sleep(wait_time)
                    
                    # Try the primary client again
                    try:
                        response = await make_groq_request(
                            primary_client, 
                            primary_index, 
                            system_prompt, 
                            user_prompt
                        )
                        
                        # Success after wait! Rotate counter
                        rotate_to_next()
                        return response
                        
                    except Exception as retry_error:
                        print(f"Retry attempt {attempt + 1} failed: {retry_error}")
                        if not is_rate_limit_error(retry_error):
                            break  # If it's not a rate limit error, no point retrying
            
            # If we get here, all attempts failed
            rotate_to_next()  # Still rotate for next request
            return "I'm temporarily unable to process your question due to high API demand. Please try again in a few moments."
    
    except Exception as e:
        print(f"Error in ask_gpt: {e}")
        return "Sorry, I couldn't process your question at the moment."