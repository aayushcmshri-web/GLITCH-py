import os
from dotenv import load_dotenv

# 1. Import ChatOpenRouter instead of ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# Verify your environment variable is loaded properly
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("Missing OPENROUTER_API_KEY in your .env file!")

# 2. Initialize the OpenRouter model wrapper
# Pass your desired OpenRouter model slug (e.g., google/gemini-2.5-flash)
# Update your LangChain model initialization line:
llm = ChatOpenRouter(
    model="google/gemini-2.5-flash",
    max_tokens=4000  # Explicitly set this to a lower value
)

# The rest of your invocation code stays completely unchanged!
response = llm.invoke("how to save a project on github")
print(response.content)