"""
RTAI Backend - Real-Time Conversational AI Backend
====================================================
A high-performance, asynchronous FastAPI backend implementing:
- Real-time WebSocket streaming with LLM
- Function/Tool calling with result integration
- Multi-turn conversation state management
- Async Supabase persistence with in-memory fallback
- Automatic post-session summary generation
- Dynamic context-aware prompt routing
"""

import os
import re
import ast
import json
import random
import asyncio
import logging
import operator
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from supabase._async.client import create_client as create_async_client, AsyncClient

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("rtai")


class Settings(BaseSettings):
    """Application configuration from environment variables."""
    supabase_url: Optional[str] = Field(default=None, description="Supabase project URL")
    supabase_key: Optional[str] = Field(default=None, description="Supabase service role key")
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    llm_model: str = Field(default="gemini-1.5-flash", description="Gemini model")
    max_tokens: int = Field(default=2048, description="Max response tokens")
    temperature: float = Field(default=0.7, description="LLM temperature")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }


def get_settings() -> Settings:
    """Lazy-load settings."""
    return Settings()


settings = get_settings()


# =============================================================================
# DATA MODELS
# =============================================================================

class EventType(str, Enum):
    """Types of events logged during a session."""
    USER_MESSAGE = "user_message"
    AI_RESPONSE = "ai_response"
    AI_CHUNK = "ai_chunk"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM_CONNECT = "system_connect"
    SYSTEM_DISCONNECT = "system_disconnect"
    SYSTEM_ERROR = "system_error"


class SessionStatus(str, Enum):
    """Session lifecycle status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SessionState:
    """In-memory session state for conversation tracking."""
    session_id: str
    user_id: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    message_count: int = 0
    active_intent: str = "general"

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        self.message_count += 1

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self.conversation_history]


# =============================================================================
# HYBRID PERSISTENCE (SUPABASE + RESILIENT IN-MEMORY STORE)
# =============================================================================

class StorageManager:
    """
    Manages persistence with async Supabase as primary,
    and a reliable in-memory store as fallback for offline or unconfigured environments.
    """
    def __init__(self):
        self.supabase_connected: bool = False
        self.client: Optional[AsyncClient] = None
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self._seq: int = 1

    async def initialize(self):
        """Initialize and verify Supabase client."""
        url = settings.supabase_url
        key = settings.supabase_key
        
        if not url or not key or "your-project" in url:
            logger.info("Supabase credentials not configured. Operating with in-memory persistence.")
            self.supabase_connected = False
            return

        try:
            self.client = await create_async_client(url, key)
            # Connectivity check with 3-second timeout
            await asyncio.wait_for(
                self.client.table("sessions").select("id").limit(1).execute(),
                timeout=3.0
            )
            self.supabase_connected = True
            logger.info("Supabase client initialized and connected successfully.")
        except Exception as e:
            logger.warning(
                f"Supabase unavailable ({type(e).__name__}: {e}). "
                "Operating with in-memory persistence fallback."
            )
            self.supabase_connected = False

    async def close(self):
        """Cleanup storage client."""
        self.client = None
        self.supabase_connected = False
        logger.info("Storage manager closed")

    async def create_session(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create or initialize a session."""
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "session_id": session_id,
            "user_id": user_id,
            "start_time": now,
            "status": SessionStatus.ACTIVE.value
        }
        self.sessions[session_id] = record
        
        if self.supabase_connected and self.client:
            try:
                res = await self.client.table("sessions").upsert(
                    record, on_conflict="session_id"
                ).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                logger.error(f"Supabase create_session error: {e}")
        return record

    async def finalize_session(self, session_id: str, summary: str, start_time: datetime) -> Dict[str, Any]:
        """Finalize a session with duration and summary."""
        end_time = datetime.now(timezone.utc)
        duration = max(0, int((end_time - start_time).total_seconds()))
        updates = {
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "summary": summary,
            "status": SessionStatus.COMPLETED.value
        }
        
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
        else:
            self.sessions[session_id] = {"session_id": session_id, **updates}

        if self.supabase_connected and self.client:
            try:
                res = await self.client.table("sessions").update(updates).eq("session_id", session_id).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                logger.error(f"Supabase finalize_session error: {e}")
        return self.sessions.get(session_id, updates)

    async def log_event(self, session_id: str, event_type: EventType, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log an event for a session."""
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "session_id": session_id,
            "event_type": event_type.value,
            "event_data": event_data,
            "timestamp": now,
            "sequence_number": self._seq
        }
        self._seq += 1

        if session_id not in self.events:
            self.events[session_id] = []
        self.events[session_id].append(record)

        if self.supabase_connected and self.client:
            try:
                res = await self.client.table("session_events").insert({
                    "session_id": session_id,
                    "event_type": event_type.value,
                    "event_data": event_data
                }).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                logger.error(f"Supabase log_event error: {e}")
        return record

    async def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve conversation history for a session."""
        if self.supabase_connected and self.client:
            try:
                res = await self.client.table("session_events")\
                    .select("*")\
                    .eq("session_id", session_id)\
                    .in_("event_type", [EventType.USER_MESSAGE.value, EventType.AI_RESPONSE.value])\
                    .order("sequence_number")\
                    .execute()
                if res.data:
                    history = []
                    for ev in res.data:
                        role = "user" if ev["event_type"] == EventType.USER_MESSAGE.value else "model"
                        history.append({"role": role, "content": ev["event_data"].get("content", "")})
                    return history
            except Exception as e:
                logger.error(f"Supabase get_conversation_history error: {e}")

        # In-memory history
        events = self.events.get(session_id, [])
        history = []
        for ev in events:
            if ev["event_type"] in [EventType.USER_MESSAGE.value, EventType.AI_RESPONSE.value]:
                role = "user" if ev["event_type"] == EventType.USER_MESSAGE.value else "model"
                history.append({"role": role, "content": ev["event_data"].get("content", "")})
        return history

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.sessions.values())[-limit:]


storage = StorageManager()

# Legacy repository adapters for backward compatibility
class SessionRepository:
    @staticmethod
    async def create(session_id: str, user_id: Optional[str] = None) -> Dict:
        return await storage.create_session(session_id, user_id)

    @staticmethod
    async def finalize(session_id: str, summary: str, start_time: datetime) -> Dict:
        return await storage.finalize_session(session_id, summary, start_time)


class EventRepository:
    @staticmethod
    async def log_event(session_id: str, event_type: EventType, event_data: Dict) -> Dict:
        return await storage.log_event(session_id, event_type, event_data)

    @staticmethod
    async def get_conversation_history(session_id: str) -> List[Dict]:
        return await storage.get_conversation_history(session_id)


async def init_supabase():
    await storage.initialize()


async def close_supabase():
    await storage.close()


def get_supabase() -> Optional[AsyncClient]:
    return storage.client


# =============================================================================
# TOOL DEFINITIONS & SAFE EXECUTOR
# =============================================================================

SAFE_MATH_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_ast_eval(node: ast.AST) -> float | int:
    """Safely evaluate AST arithmetic expressions without security or DoS risks."""
    if isinstance(node, ast.Expression):
        return _safe_ast_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant type")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_MATH_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_ast_eval(node.left)
        right = _safe_ast_eval(node.right)
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("division by zero")
        return SAFE_MATH_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_MATH_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return SAFE_MATH_OPERATORS[op_type](_safe_ast_eval(node.operand))
    raise ValueError("Invalid mathematical syntax")


async def calculate(expression: str) -> Dict[str, Any]:
    """Safe mathematical calculator with AST verification."""
    logger.info(f"Tool: calculate({expression})")
    expr_clean = expression.strip()
    if not expr_clean:
        return {"expression": expression, "error": "Empty expression", "result": None}
    if len(expr_clean) > 200:
        return {"expression": expression, "error": "Expression exceeds maximum length", "result": None}
    
    # Explicitly prohibit exponentiation to prevent CPU hang / DoS
    if "**" in expr_clean or "^" in expr_clean:
        return {"expression": expression, "error": "Exponentiation is disabled for safety", "result": None}

    try:
        tree = ast.parse(expr_clean, mode="eval")
        result = _safe_ast_eval(tree)
        # Format whole floats cleanly (e.g. 5.0 -> 5)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return {"expression": expression, "result": result}
    except ZeroDivisionError:
        return {"expression": expression, "error": "Division by zero", "result": None}
    except Exception as e:
        return {"expression": expression, "error": f"Invalid expression: {str(e)}", "result": None}


async def get_weather(city: str, unit: str = "celsius") -> Dict[str, Any]:
    """Simulated weather API call with realistic values."""
    logger.info(f"Tool: get_weather({city}, {unit})")
    unit_norm = unit.lower().strip()
    if unit_norm not in ["celsius", "fahrenheit"]:
        unit_norm = "celsius"

    city_clean = city.strip().title()
    seed_val = sum(ord(c) for c in city_clean) if city_clean else 42
    conditions = ["Sunny", "Partly Cloudy", "Clear", "Overcast", "Light Rain", "Breezy"]
    condition = conditions[seed_val % len(conditions)]
    temp_c = 15 + (seed_val % 16)
    temp = temp_c if unit_norm == "celsius" else round((temp_c * 9 / 5) + 32, 1)
    humidity = 35 + (seed_val % 50)

    return {
        "city": city_clean,
        "temperature": temp,
        "unit": unit_norm,
        "condition": condition,
        "humidity": humidity,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


async def search_database(query: str, limit: int = 5) -> Dict[str, Any]:
    """Simulated database search with catalog search."""
    logger.info(f"Tool: search_database({query}, limit={limit})")
    limit = max(1, min(limit or 5, 20))
    catalog = [
        {"id": 1, "title": "Introduction to Machine Learning", "category": "AI", "snippet": "Core neural network and ML fundamentals."},
        {"id": 2, "title": "Python Best Practices", "category": "Programming", "snippet": "Type hints, async/await patterns, and code structure."},
        {"id": 3, "title": "Database Optimization", "category": "Database", "snippet": "PostgreSQL indexing, query plans, and scaling."},
        {"id": 4, "title": "Web Development with FastAPI", "category": "Web", "snippet": "Building async APIs with Pydantic and WebSockets."},
        {"id": 5, "title": "Cloud Architecture Patterns", "category": "Cloud", "snippet": "Microservices, resilience, and event streaming."},
        {"id": 6, "title": "Real-Time WebSocket Protocol", "category": "Networking", "snippet": "Full-duplex bi-directional streaming protocols."},
        {"id": 7, "title": "Prompt Engineering & Function Calling", "category": "AI", "snippet": "LLM tool calling, schema declaration, and routing."}
    ]
    q_lower = query.lower().strip()
    filtered = [
        item for item in catalog
        if q_lower in item["title"].lower() or q_lower in item["category"].lower() or q_lower in item["snippet"].lower()
    ]
    results = (filtered if filtered else catalog)[:limit]
    return {"query": query, "results": results, "total": len(results)}


async def get_current_time(timezone_name: str = "UTC") -> Dict[str, Any]:
    """Get current date and time."""
    logger.info(f"Tool: get_current_time({timezone_name})")
    tz_str = (timezone_name or "UTC").strip()
    tz = timezone.utc
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_str)
    except Exception:
        tz = timezone.utc
        tz_str = "UTC"

    now = datetime.now(tz)
    return {
        "timezone": tz_str,
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A")
    }


TOOLS: Dict[str, Callable] = {
    "get_weather": get_weather,
    "search_database": search_database,
    "calculate": calculate,
    "get_current_time": get_current_time
}

TOOL_DECLARATIONS = [
    genai.protos.Tool(function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_weather",
            description="Get current weather for a city",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "city": genai.protos.Schema(type=genai.protos.Type.STRING, description="City name"),
                    "unit": genai.protos.Schema(type=genai.protos.Type.STRING, enum=["celsius", "fahrenheit"])
                },
                required=["city"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="search_database",
            description="Search database for documentation and articles",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "query": genai.protos.Schema(type=genai.protos.Type.STRING, description="Search query"),
                    "limit": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Max results")
                },
                required=["query"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="calculate",
            description="Perform mathematical calculations",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "expression": genai.protos.Schema(type=genai.protos.Type.STRING, description="Math expression")
                },
                required=["expression"]
            )
        ),
        genai.protos.FunctionDeclaration(
            name="get_current_time",
            description="Get current date and time",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "timezone_name": genai.protos.Schema(type=genai.protos.Type.STRING, description="Timezone name")
                },
                required=[]
            )
        )
    ])
]


# =============================================================================
# PROMPT MANAGEMENT & ROUTING
# =============================================================================

class PromptManager:
    """Context-aware system prompt routing."""

    BASE_PROMPT = """You are ARIA, a knowledgeable and helpful AI assistant.

IMPORTANT FORMATTING RULES:
- NEVER use asterisks (*) or bullet points
- ALWAYS use numbered lists (1., 2., 3.) for multiple items
- Use simple, clear paragraphs for explanations
- Keep responses clean and professionally formatted

You have optional tools when needed:
- get_weather: Real-time weather data
- search_database: Look up information
- calculate: Complex math
- get_current_time: Current date/time

Answer using your general knowledge. Use tools only when absolutely necessary for real-time data."""

    WEATHER_PROMPT = """You are ARIA, a weather specialist. Use get_weather for accurate data.
Provide context about conditions and practical advice."""

    MATH_PROMPT = """You are ARIA, a math specialist. Use calculate for precise results.
Show work step-by-step and verify calculations."""

    TECHNICAL_PROMPT = """You are ARIA, a technical expert. Use search_database for documentation.
Provide detailed explanations with best practices."""

    TIME_PROMPT = """You are ARIA, a helpful assistant with time awareness. Use get_current_time for real-time date and time details."""

    SUMMARY_PROMPT = """Generate a concise 2-3 sentence summary of this conversation.
Identify main topics, key questions, tools used, and outcomes."""

    INTENT_PATTERNS = {
        "weather": [r"\bweather\b", r"\btemperature\b", r"\brain\b", r"\bsunn?y\b", r"\bforecast\b", r"\bclimate\b", r"\bhumidity\b"],
        "math": [r"\bcalculat\w*\b", r"\bcompute\b", r"\bmath\b", r"\b\d+\s*[\+\-\*\/%]\s*\d+\b", r"\bsum\b", r"\bmultiply\b", r"\bdivide\b"],
        "technical": [r"\bcode\b", r"\bprogram\w*\b", r"\bapi\b", r"\bdatabase\b", r"\bpython\b", r"\bquery\b", r"\btutorial\b"],
        "time": [r"\btime\b", r"\bclock\b", r"\bdate\b", r"\btoday\b", r"\bday of week\b"]
    }

    @classmethod
    def detect_intent(cls, message: str) -> str:
        msg_lower = message.lower()
        for intent, patterns in cls.INTENT_PATTERNS.items():
            if any(re.search(p, msg_lower) for p in patterns):
                return intent
        return "general"

    @classmethod
    def get_routed_prompt(cls, message: str) -> str:
        intent = cls.detect_intent(message)
        return {
            "weather": cls.WEATHER_PROMPT,
            "math": cls.MATH_PROMPT,
            "technical": cls.TECHNICAL_PROMPT,
            "time": cls.TIME_PROMPT
        }.get(intent, cls.BASE_PROMPT)


# =============================================================================
# LLM CLIENT & ARIA CONVERSATIONAL ENGINE
# =============================================================================

class LLMClient:
    """
    LLM Client supporting both live Google Gemini API streaming
    and a high-performance ARIA conversational fallback engine
    when Gemini credentials are not configured or offline.
    """
    def __init__(self):
        self.chat_sessions: Dict[str, Any] = {}
        self.use_gemini: bool = False
        
        api_key = settings.google_api_key
        if api_key and not api_key.startswith("your-") and len(api_key) > 20:
            try:
                genai.configure(api_key=api_key)
                self.use_gemini = True
                logger.info("Gemini API configured")
            except Exception as e:
                logger.warning(f"Gemini configuration error: {e}. Using local conversational engine.")
                self.use_gemini = False
        else:
            logger.info("Gemini API key not configured. Using ARIA conversational engine.")
            self.use_gemini = False

    def _get_model(self, system_prompt: str):
        return genai.GenerativeModel(
            model_name=settings.llm_model,
            generation_config=GenerationConfig(
                temperature=settings.temperature,
                max_output_tokens=settings.max_tokens
            ),
            system_instruction=system_prompt,
            tools=TOOL_DECLARATIONS
        )

    def get_or_create_chat(self, session_id: str, history: List[Dict] = None, system_prompt: str = None):
        if session_id not in self.chat_sessions:
            model = self._get_model(system_prompt or PromptManager.BASE_PROMPT)
            formatted = [{"role": m["role"], "parts": [m["content"]]} for m in (history or [])]
            self.chat_sessions[session_id] = model.start_chat(history=formatted)
        return self.chat_sessions[session_id]

    def clear_chat(self, session_id: str):
        self.chat_sessions.pop(session_id, None)

    async def generate_streaming(
        self,
        session_id: str,
        message: str,
        history: List[Dict] = None,
        on_tool_call: Callable = None,
        on_tool_result: Callable = None
    ):
        """Generate streaming response with dynamic prompt routing and tool calling."""
        routed_prompt = PromptManager.get_routed_prompt(message)
        intent = PromptManager.detect_intent(message)

        if self.use_gemini:
            try:
                chat = self.get_or_create_chat(session_id, history, routed_prompt)
                response = await asyncio.to_thread(chat.send_message, message, stream=True)
                
                function_call = None
                for chunk in response:
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                function_call = part.function_call
                            elif hasattr(part, 'text') and part.text:
                                yield part.text
                
                if function_call:
                    tool_name = function_call.name
                    tool_args = dict(function_call.args)
                    
                    if on_tool_call:
                        await on_tool_call(tool_name, tool_args)
                    
                    if tool_name in TOOLS:
                        try:
                            tool_result = await TOOLS[tool_name](**tool_args)
                        except Exception as te:
                            tool_result = {"error": str(te)}
                    else:
                        tool_result = {"error": f"Unknown tool: {tool_name}"}

                    if on_tool_result:
                        await on_tool_result(tool_name, tool_result)

                    tool_response = await asyncio.to_thread(
                        chat.send_message,
                        genai.protos.Content(parts=[
                            genai.protos.Part(function_response=genai.protos.FunctionResponse(
                                name=tool_name, response={"result": tool_result}
                            ))
                        ]),
                        stream=True
                    )

                    for chunk in tool_response:
                        if hasattr(chunk, 'candidates') and chunk.candidates:
                            for part in chunk.candidates[0].content.parts:
                                if hasattr(part, 'text') and part.text:
                                    yield part.text
                return
            except Exception as e:
                logger.warning(f"Gemini API streaming error: {e}. Switching to ARIA engine.")
                self.use_gemini = False

        # ARIA Conversational Engine Fallback
        async for chunk in self._fallback_streaming(message, intent, on_tool_call, on_tool_result):
            yield chunk

    async def _fallback_streaming(
        self,
        message: str,
        intent: str,
        on_tool_call: Callable,
        on_tool_result: Callable
    ):
        """Simulated real-time streaming with tool calling and intent-driven logic."""
        response_text = ""

        if intent == "weather":
            city_match = re.search(r"\b(?:in|for|at)\b\s+([a-zA-Z\s]+?)(?:\?|\.|$)", message, re.IGNORECASE)
            city = city_match.group(1).strip() if city_match else "Tokyo"
            tool_args = {"city": city, "unit": "celsius"}
            
            if on_tool_call:
                await on_tool_call("get_weather", tool_args)
            result = await get_weather(**tool_args)
            if on_tool_result:
                await on_tool_result("get_weather", result)

            response_text = (
                f"Here is the real-time weather update for {result['city']}:\n\n"
                f"1. Current Condition: {result['condition']}\n"
                f"2. Temperature: {result['temperature']}°C\n"
                f"3. Humidity: {result['humidity']}%\n\n"
                f"Conditions are favorable. Let me know if you would like a forecast for any other city."
            )

        elif intent == "math":
            expr_match = re.search(r"(\d+[\d\.\s\+\-\*\/\(\)\%]*\d+|\d+)", message)
            expr = expr_match.group(1).strip() if expr_match else "125 * 48 + 375"
            tool_args = {"expression": expr}
            
            if on_tool_call:
                await on_tool_call("calculate", tool_args)
            result = await calculate(**tool_args)
            if on_tool_result:
                await on_tool_result("calculate", result)

            if result.get("error"):
                response_text = f"I attempted to evaluate the expression '{expr}', but encountered an issue: {result['error']}."
            else:
                response_text = (
                    f"Calculation details for: {result['expression']}\n\n"
                    f"1. Parsed expression: {result['expression']}\n"
                    f"2. Computed result: {result['result']}\n\n"
                    f"The verified answer is {result['result']}."
                )

        elif intent == "technical":
            query_match = re.search(r"\b(?:search|for|tutorials?|find|about)\b\s+([a-zA-Z0-9\s]+?)(?:\?|\.|$)", message, re.IGNORECASE)
            query = query_match.group(1).strip() if query_match else "Python"
            tool_args = {"query": query, "limit": 4}
            
            if on_tool_call:
                await on_tool_call("search_database", tool_args)
            result = await search_database(**tool_args)
            if on_tool_result:
                await on_tool_result("search_database", result)

            items_text = "\n".join(
                f"{idx + 1}. {item['title']} ({item['category']}): {item['snippet']}"
                for idx, item in enumerate(result.get("results", []))
            )
            response_text = (
                f"I searched the technical knowledge base for '{result['query']}' and retrieved {len(result.get('results', []))} relevant entries:\n\n"
                f"{items_text}\n\n"
                f"Let me know if you would like me to elaborate on any of these topics."
            )

        elif intent == "time":
            tool_args = {"timezone_name": "UTC"}
            if on_tool_call:
                await on_tool_call("get_current_time", tool_args)
            result = await get_current_time(**tool_args)
            if on_tool_result:
                await on_tool_result("get_current_time", result)

            response_text = (
                f"Current temporal data:\n\n"
                f"1. Date: {result['date']}\n"
                f"2. Time: {result['time']} ({result['timezone']})\n"
                f"3. Day of week: {result['day_of_week']}"
            )

        else:
            response_text = (
                "Hello! I am ARIA, your high-performance conversational assistant.\n\n"
                "I can assist you with real-time operations including:\n"
                "1. Weather reports for any city worldwide\n"
                "2. Mathematical calculations and expressions\n"
                "3. Searching technical documentation and knowledge bases\n"
                "4. Live date and time lookups\n\n"
                "How can I assist you today?"
            )

        # Stream words with micro-delays for realistic streaming
        words = re.findall(r"\S+|\n", response_text)
        for i, word in enumerate(words):
            if word == "\n":
                yield "\n"
            else:
                yield word + (" " if i < len(words) - 1 and words[i + 1] != "\n" else "")
            await asyncio.sleep(0.015)

    async def generate_summary(self, history: List[Dict]) -> str:
        """Generate conversation summary."""
        if not history:
            return "No conversation recorded."

        if self.use_gemini:
            try:
                model = genai.GenerativeModel(
                    model_name=settings.llm_model,
                    generation_config=GenerationConfig(temperature=0.3, max_output_tokens=300),
                    system_instruction=PromptManager.SUMMARY_PROMPT
                )
                conv = "\n".join(f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in history)
                res = await asyncio.to_thread(model.generate_content, f"Summarize:\n\n{conv}")
                if res and res.text:
                    return res.text.strip()
            except Exception as e:
                logger.warning(f"Gemini summary generation error: {e}")

        # Local summary generator
        user_msgs = [m["content"] for m in history if m["role"] == "user"]
        msg_count = len(history)
        sample_query = user_msgs[0] if user_msgs else "general assistance"
        return f"Session completed with {msg_count} turns. Topics discussed centered on '{sample_query[:50]}'. Tools and responses were delivered successfully."


llm_client = LLMClient()
sessions: Dict[str, SessionState] = {}


# =============================================================================
# POST-SESSION PROCESSING
# =============================================================================

async def process_session_end(session_id: str, start_time: datetime, in_memory_history: List[Dict]):
    """Background task: generate summary and finalize session."""
    logger.info(f"Post-processing session: {session_id}")
    try:
        history = await storage.get_conversation_history(session_id)
        if not history and in_memory_history:
            history = in_memory_history

        summary = await llm_client.generate_summary(history)
        await storage.finalize_session(session_id, summary, start_time)
        logger.info(f"Session {session_id} finalized with summary: {summary[:60]}...")
    except Exception as e:
        logger.error(f"Post-processing error for {session_id}: {e}")
        try:
            await storage.finalize_session(session_id, "Session finalized with partial logs.", start_time)
        except Exception:
            pass


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("Starting RTAI Backend...")
    await storage.initialize()
    yield
    await storage.close()
    logger.info("RTAI Backend shutdown complete")


app = FastAPI(
    title="RTAI Backend",
    description="Real-Time Conversational AI Backend with WebSocket Streaming and Persistence",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)


@app.get("/")
async def root():
    """API root with service info and links."""
    return {
        "name": "RTAI Backend",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "chat_ui": "/chat",
            "sessions": "/sessions",
            "websocket": "/ws/session/{session_id}"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint with subsystem status."""
    return {
        "status": "healthy",
        "service": "rtai-backend",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "persistence": "supabase" if storage.supabase_connected else "in_memory_fallback",
        "llm_engine": "gemini" if llm_client.use_gemini else "local_conversational_engine",
        "active_sessions": len(sessions)
    }


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """Serve the ARIA Chat UI client."""
    client_path = Path(__file__).parent / "tests" / "test_client.html"
    if client_path.exists():
        return FileResponse(str(client_path), media_type="text/html")
    return HTMLResponse("<h3>Chat client file not found.</h3>", status_code=404)


@app.get("/sessions")
async def list_sessions(limit: int = 50):
    """List recent sessions."""
    return {"sessions": storage.list_sessions(limit)}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get details and event history for a specific session."""
    session = storage.get_session(session_id)
    history = await storage.get_conversation_history(session_id)
    raw_events = storage.events.get(session_id, [])
    if not session and not raw_events:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session": session or {"session_id": session_id, "status": "unknown"},
        "history": history,
        "events": raw_events
    }


@app.websocket("/ws/session/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time conversational sessions.

    Protocol:
        Client -> Server: {"type": "user_message", "content": "..."}
        Client -> Server: {"type": "ping"}
        Server -> Client: {"type": "pong"}
        Server -> Client: {"type": "system", "content": "..."}
        Server -> Client: {"type": "ai_chunk", "content": "..."}
        Server -> Client: {"type": "ai_complete", "content": "..."}
        Server -> Client: {"type": "tool_call", "data": {...}}
        Server -> Client: {"type": "tool_result", "data": {...}}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    user_id = websocket.query_params.get("user_id")

    # Initialize or restore session state
    session_state = SessionState(session_id=session_id, user_id=user_id)
    sessions[session_id] = session_state

    # Persist session creation
    try:
        await storage.create_session(session_id, user_id)
        await storage.log_event(
            session_id,
            EventType.SYSTEM_CONNECT,
            {"user_id": user_id, "timestamp": datetime.now(timezone.utc).isoformat()}
        )
    except Exception as e:
        logger.error(f"Storage error on connect: {e}")

    # Send welcome message
    await websocket.send_json(WebSocketMessage(
        type="system",
        content="Connected to ARIA. How can I help you?",
        data={"session_id": session_id}
    ).model_dump())

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "user_message")
            content = data.get("content", "")

            # Heartbeat ping/pong
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type != "user_message" or not content or not content.strip():
                continue

            content = content.strip()
            logger.info(f"[{session_id}] User: {content[:80]}...")

            session_state.add_message("user", content)
            session_state.active_intent = PromptManager.detect_intent(content)

            try:
                await storage.log_event(session_id, EventType.USER_MESSAGE, {"content": content})
            except Exception as e:
                logger.error(f"DB log error: {e}")

            full_response = ""

            async def on_tool_call(name: str, args: Dict):
                await websocket.send_json(WebSocketMessage(
                    type="tool_call", data={"tool": name, "arguments": args}
                ).model_dump())
                try:
                    await storage.log_event(
                        session_id,
                        EventType.TOOL_CALL,
                        {"tool": name, "arguments": args}
                    )
                except Exception:
                    pass

            async def on_tool_result(name: str, result: Dict):
                await websocket.send_json(WebSocketMessage(
                    type="tool_result", data={"tool": name, "result": result}
                ).model_dump())
                try:
                    await storage.log_event(
                        session_id,
                        EventType.TOOL_RESULT,
                        {"tool": name, "result": result}
                    )
                except Exception:
                    pass

            try:
                history = session_state.get_history_for_llm()[:-1] if session_state.message_count > 1 else None

                async for chunk in llm_client.generate_streaming(
                    session_id, content, history, on_tool_call, on_tool_result
                ):
                    full_response += chunk
                    await websocket.send_json(WebSocketMessage(
                        type="ai_chunk", content=chunk
                    ).model_dump())

                # Send complete message
                await websocket.send_json(WebSocketMessage(
                    type="ai_complete", content=full_response
                ).model_dump())

                session_state.add_message("model", full_response)

                try:
                    await storage.log_event(
                        session_id,
                        EventType.AI_RESPONSE,
                        {"content": full_response}
                    )
                except Exception:
                    pass

                logger.info(f"[{session_id}] AI: {full_response[:80]}...")

            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                await websocket.send_json(WebSocketMessage(
                    type="error",
                    content=f"An error occurred while generating the response: {str(e)}"
                ).model_dump())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket unexpected error: {e}")
    finally:
        try:
            await storage.log_event(
                session_id,
                EventType.SYSTEM_DISCONNECT,
                {"timestamp": datetime.now(timezone.utc).isoformat()}
            )
        except Exception:
            pass

        # Trigger post-session summary background task
        if session_state.message_count > 0:
            asyncio.create_task(process_session_end(
                session_id,
                session_state.start_time,
                session_state.conversation_history.copy()
            ))

        sessions.pop(session_id, None)
        llm_client.clear_chat(session_id)
        logger.info(f"Session cleanup complete: {session_id}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
