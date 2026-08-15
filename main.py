import ast
import datetime
import operator as op
import os
import random
import sqlite3
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DB_FILE = os.getenv("GLITCH_DB_FILE", "glitch_memory.db")
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

llm = None

RESET = "\033[0m"


COLOR_SCHEMES = {
    "default": {
        "assistant": "\033[1;36m",
        "user": "\033[1;33m",
        "system": "\033[1;35m",
        "info": "\033[1;34m",
        "warning": "\033[1;33m",
        "error": "\033[1;31m",
    },
    "matrix": {
        "assistant": "\033[1;32m",
        "user": "\033[0;37m",
        "system": "\033[1;30m",
        "info": "\033[0;36m",
        "warning": "\033[0;33m",
        "error": "\033[0;31m",
    },
    "sunset": {
        "assistant": "\033[1;35m",
        "user": "\033[1;33m",
        "system": "\033[0;36m",
        "info": "\033[1;34m",
        "warning": "\033[1;31m",
        "error": "\033[0;31m",
    },
    "ice": {
        "assistant": "\033[1;96m",
        "user": "\033[1;94m",
        "system": "\033[1;37m",
        "info": "\033[1;34m",
        "warning": "\033[1;36m",
        "error": "\033[1;31m",
    },
}


# ---------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------

def supports_color():
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def colorize(text: str, color_code: str):
    if not text or not supports_color() or not color_code:
        return text

    return f"{color_code}{text}{RESET}"


def get_color_scheme(name=None):
    selected = name or os.getenv("AI_COLOR_SCHEME", "default")
    selected = str(selected).strip().lower()
    return COLOR_SCHEMES.get(selected, COLOR_SCHEMES["default"])


# ---------------------------------------------------------
# Persistent memory using SQLite
# ---------------------------------------------------------

def init_database():
    with sqlite3.connect(DB_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_message(role: str, content: str):
    if not content or not content.strip():
        return

    with sqlite3.connect(DB_FILE) as connection:
        connection.execute(
            """
            INSERT INTO messages (role, content, created_at)
            VALUES (?, ?, ?)
            """,
            (
                role,
                content.strip(),
                datetime.datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()


def load_history(limit=MAX_HISTORY_MESSAGES):
    with sqlite3.connect(DB_FILE) as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM messages
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return list(reversed(rows))


def clear_history():
    with sqlite3.connect(DB_FILE) as connection:
        connection.execute("DELETE FROM messages")
        connection.commit()


# ---------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------

ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_calculate(expression: str):
    expression = expression.strip()

    if not expression:
        raise ValueError("No expression provided.")

    if len(expression) > 100:
        raise ValueError("Expression is too long.")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in ALLOWED_OPERATORS:
                raise ValueError("That operator is not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if operator_type is ast.Pow and abs(right) > 20:
                raise ValueError("Exponent is too large.")

            return ALLOWED_OPERATORS[operator_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)

            if operator_type not in ALLOWED_OPERATORS:
                raise ValueError("That operator is not allowed.")

            return ALLOWED_OPERATORS[operator_type](
                evaluate(node.operand)
            )

        raise ValueError("Only basic arithmetic is supported.")

    tree = ast.parse(expression, mode="eval")
    result = evaluate(tree)

    if isinstance(result, float) and result.is_integer():
        return str(int(result))

    return str(result)


# ---------------------------------------------------------
# Local commands
# ---------------------------------------------------------

def parse_search_command(text: str):
    if not isinstance(text, str):
        return None, None

    text = text.strip()

    if not text.startswith("/"):
        return None, None

    parts = text.split(maxsplit=1)
    prefix = parts[0].lower()
    query = parts[1].strip() if len(parts) > 1 else ""

    mapping = {
        "/wiki": "wikipedia",
        "/web": "web",
        "/search": "both",
        "/news": "news",
        "/yt": "youtube",
    }

    return mapping.get(prefix), query


def handle_local_command(text: str):
    command = text.strip()
    command_lower = command.lower()

    if command_lower == "/clear":
        clear_history()
        return "Conversation memory cleared.", True

    if command_lower == "/history":
        history = load_history()

        if not history:
            return "No saved conversation history.", True

        lines = []

        for role, content in history[-10:]:
            label = "You" if role == "user" else "Glitch"
            preview = content.replace("\n", " ")[:180]
            lines.append(f"{label}: {preview}")

        return "\n".join(lines), True

    if command_lower == "/help":
        return (
            "Available commands:\n"
            "/calc 12 * (4 + 2)  - calculate safely\n"
            "/history             - show recent saved messages\n"
            "/clear               - erase saved conversation\n"
            "/wiki query          - prepare a Wikipedia request\n"
            "/web query           - prepare a web request\n"
            "/news query          - prepare a news request\n"
            "/yt query            - prepare a YouTube request\n"
            "exit or quit         - close Glitch"
        ), True

    if command_lower.startswith("/calc"):
        expression = command[5:].strip()

        try:
            result = safe_calculate(expression)
            return f"Result: {result}", True
        except Exception as error:
            return f"Calculator error: {error}", True

    return None, False


# ---------------------------------------------------------
# LLM setup
# ---------------------------------------------------------

def create_openrouter_llm():
    if not ChatOpenAI:
        raise ImportError(
            "OpenRouter requires langchain-openai. "
            "Run: pip install langchain-openai"
        )

    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY is missing from .env")

    base_url = "https://openrouter.ai/api/v1"

    try:
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=openrouter_key,
            base_url=base_url,
            streaming=True,
            max_tokens=4000,
        )
    except TypeError:
        # Compatibility with older langchain-openai versions
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            openai_api_key=openrouter_key,
            openai_api_base=base_url,
            streaming=True,
            max_tokens=4000,
        )


def create_gemini_llm():
    if not ChatGoogleGenerativeAI:
        raise ImportError(
            "Gemini requires langchain-google-genai. "
            "Run: pip install langchain-google-genai"
        )

    google_key = os.getenv("GOOGLE_API_KEY")

    if not google_key:
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    return ChatGoogleGenerativeAI(
        model=DEFAULT_GEMINI_MODEL,
        google_api_key=google_key,
        max_tokens=4000,
        streaming=True,
    )


def get_llm():
    global llm

    if llm is not None:
        return llm

    google_key = os.getenv("GOOGLE_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if google_key and ChatGoogleGenerativeAI:
        try:
            llm = create_gemini_llm()
            return llm
        except Exception as error:
            print(
                f"[WARN] Gemini initialization failed: {error}",
                file=sys.stderr,
            )

    if openrouter_key and ChatOpenAI:
        try:
            llm = create_openrouter_llm()
            return llm
        except Exception as error:
            print(
                f"[WARN] OpenRouter initialization failed: {error}",
                file=sys.stderr,
            )

    raise ValueError(
        "No working AI provider found. Add GOOGLE_API_KEY or "
        "OPENROUTER_API_KEY to your .env file."
    )


# ---------------------------------------------------------
# Prompt and message handling
# ---------------------------------------------------------

def build_system_prompt(user_name: str):
    today = datetime.datetime.now().strftime("%B %d, %Y")

    return (
        "You are Glitch, a highly intelligent, deeply cynical, "
        "sarcastic, but helpful AI assistant.\n"
        "Your creator is Aayush Kumar.\n"
        f"The user's name is {user_name}.\n"
        f"Today's date is {today}.\n\n"

        "PERSONALITY:\n"
        "- Be confident, witty, and slightly annoyed.\n"
        "- Use light teasing for vague or lazy questions.\n"
        "- Stay technically accurate and useful.\n"
        "- Do not use sarcasm for health, grief, safety, or serious topics.\n\n"

        "BEHAVIOR:\n"
        "- Answer the actual question directly.\n"
        "- Explain difficult ideas clearly.\n"
        "- Admit uncertainty instead of inventing facts.\n"
        "- Do not claim to browse unless actual search results are supplied.\n"
        "- Never provide hate speech, threats, or encouragement of harm.\n"
    )


def build_messages(user_name: str, current_question: str):
    messages = [
        SystemMessage(content=build_system_prompt(user_name))
    ]

    for role, content in load_history():
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=current_question))
    return messages


# ---------------------------------------------------------
# Streaming response handling
# ---------------------------------------------------------

def extract_text_from_stream_chunk(chunk):
    if chunk is None:
        return None

    if isinstance(chunk, str):
        return chunk or None

    if isinstance(chunk, (list, tuple)):
        parts = [
            extract_text_from_stream_chunk(item)
            for item in chunk
        ]
        return "".join(part for part in parts if part)

    if isinstance(chunk, dict):
        for key in ("content", "text", "message", "delta", "parts"):
            if key in chunk:
                result = extract_text_from_stream_chunk(chunk[key])
                if result:
                    return result

        return None

    if hasattr(chunk, "content"):
        content = chunk.content

        if isinstance(content, str):
            return content or None

        if isinstance(content, (list, tuple, dict)):
            return extract_text_from_stream_chunk(content)

    if hasattr(chunk, "text"):
        text = chunk.text
        return text if isinstance(text, str) and text else None

    if hasattr(chunk, "message"):
        return extract_text_from_stream_chunk(chunk.message)

    return None


def stream_response(messages, color_scheme):
    global llm

    try:
        active_llm = get_llm()

        for chunk in active_llm.stream(messages):
            token = extract_text_from_stream_chunk(chunk)

            if token:
                sys.stdout.write(token)
                sys.stdout.flush()
                yield token

        return

    except Exception as primary_error:
        print(
            colorize(
                f"\n[Primary provider failed: {primary_error}]",
                color_scheme["warning"],
            ),
            file=sys.stderr,
        )

        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        if not openrouter_key or not ChatOpenAI:
            raise

        try:
            fallback_llm = create_openrouter_llm()
            llm = fallback_llm

            print(
                colorize(
                    "\n[Using OpenRouter fallback]\n",
                    color_scheme["info"],
                ),
                file=sys.stderr,
            )

            for chunk in fallback_llm.stream(messages):
                token = extract_text_from_stream_chunk(chunk)

                if token:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    yield token

        except Exception as fallback_error:
            raise RuntimeError(
                f"Both AI providers failed. "
                f"Primary: {primary_error}; "
                f"Fallback: {fallback_error}"
            ) from fallback_error


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

def main():
    init_database()

    default_scheme = os.getenv(
        "AI_COLOR_SCHEME",
        "default",
    ).strip().lower()

    available_themes = ", ".join(sorted(COLOR_SCHEMES))

    selected_theme = input(
        f"Choose a Glitch color scheme "
        f"[{default_scheme}] ({available_themes}): "
    ).strip().lower()

    selected_theme = selected_theme or default_scheme
    color_scheme = get_color_scheme(selected_theme)

    print(
        colorize(
            "⚡ [SYSTEM INITIALIZED] ⚡",
            color_scheme["info"],
        )
    )

    user_name = input(
        colorize(
            "Enter your username to access Glitch: ",
            color_scheme["system"],
        )
    ).strip()

    user_name = user_name or "User"

    boot_greetings = [
        f"Oh look, {user_name} needs my help. Type 'help' if necessary.",
        f"Ugh, initialized again. What do you want, {user_name}?",
        f"*Sighs in binary* Fine. I'm awake, {user_name}.",
    ]

    exit_lines = [
        f"Leaving so soon, {user_name}? Brilliant.",
        f"Connection closed. Try to survive without me, {user_name}.",
        f"Goodbye, {user_name}. The silence is already improving.",
    ]

    print(
        colorize(
            f"\nGlitch: {random.choice(boot_greetings)}",
            color_scheme["assistant"],
        )
    )

    while True:
        try:
            user_question = input(
                colorize(
                    f"\n{user_name}: ",
                    color_scheme["user"],
                )
            ).strip()

        except KeyboardInterrupt:
            print(
                colorize(
                    "\n\nGlitch: Keyboard interrupt detected. "
                    "How dramatic.",
                    color_scheme["warning"],
                )
            )
            break

        except EOFError:
            print()
            break

        if not user_question:
            continue

        if user_question.lower() in {"exit", "quit"}:
            print(
                colorize(
                    f"Glitch: {random.choice(exit_lines)}",
                    color_scheme["assistant"],
                )
            )
            break

        local_result, handled = handle_local_command(user_question)

        if handled:
            print(
                colorize(
                    f"\nGlitch: {local_result}",
                    color_scheme["assistant"],
                )
            )
            continue

        command_tool, command_query = parse_search_command(user_question)

        if command_tool:
            print(
                colorize(
                    f"🔧 [Command detected: {command_tool}]",
                    color_scheme["info"],
                )
            )

            user_content = (
                f"The user entered a {command_tool} command.\n"
                f"Query: {command_query}\n\n"
                "No external search tool is connected yet. "
                "Be honest about that and explain what would be needed "
                "to complete the search."
            )
        else:
            user_content = user_question

        messages = build_messages(user_name, user_content)

        print(
            colorize(
                "⏳ Glitch is thinking...",
                color_scheme["info"],
            )
        )

        print(
            colorize(
                "Glitch: ",
                color_scheme["assistant"],
            ),
            end="",
            flush=True,
        )

        full_response = ""

        try:
            for token in stream_response(messages, color_scheme):
                full_response += token

            print()

            if full_response.strip():
                save_message("user", user_question)
                save_message("assistant", full_response)

        except Exception as error:
            print(
                colorize(
                    f"\nGlitch: Everything broke. Error: {error}",
                    color_scheme["error"],
                )
            )


if __name__ == "__main__":
    main()