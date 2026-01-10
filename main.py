"""
RTAI Backend - Real-Time Conversational AI Backend
====================================================
A high-performance, asynchronous FastAPI backend implementing:
- Real-time WebSocket streaming with LLM
- Function/Tool calling with result integration
- Multi-turn conversation state management
- Async Supabase persistence (sessions + events)
- Automatic post-session summary generation

Requirements:
    pip install fastapi uvicorn google-generativeai supabase python-dotenv pydantic-settings

Environment Variables (.env):
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_KEY=your-service-role-key
    GOOGLE_API_KEY=your-gemini-api-key
"""

import os
import re
import json
import random
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from supabase import create_client, Client
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
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_key: str = Field(..., description="Supabase service role key")
    google_api_key: str = Field(..., description="Google Gemini API key")
    llm_model: str = Field(default="gemini-2.5-flash", description="Gemini model")
    max_tokens: int = Field(default=2048, description="Max response tokens")
    temperature: float = Field(default=0.7, description="LLM temperature")

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
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SessionState:
    """In-memory session state for conversation tracking."""
    session_id: str
    user_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    message_count: int = 0

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        self.message_count += 1

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self.conversation_history]


# =============================================================================
# SUPABASE CLIENT
# =============================================================================

_supabase_client: Optional[AsyncClient] = None


async def init_supabase() -> AsyncClient:
    """Initialize async Supabase client."""
    global _supabase_client
    _supabase_client = await create_async_client(settings.supabase_url, settings.supabase_key)
    logger.info("Supabase client initialized")
    return _supabase_client


async def close_supabase():
    """Cleanup Supabase client."""
    global _supabase_client
    _supabase_client = None
    logger.info("Supabase client closed")


def get_supabase() -> AsyncClient:
    """Get the Supabase client instance."""
    if _supabase_client is None:
        raise RuntimeError("Supabase not initialized")
    return _supabase_client


# =============================================================================
# DATABASE REPOSITORIES
# =============================================================================

class SessionRepository:
    """CRUD operations for sessions table."""
    
    @staticmethod
    async def create(session_id: str, user_id: Optional[str] = None) -> Dict:
        client = get_supabase()
        result = await client.table("sessions").insert({
            "session_id": session_id,
            "user_id": user_id,
            "status": SessionStatus.ACTIVE.value
        }).execute()
        logger.info(f"Session created: {session_id}")
        return result.data[0] if result.data else {}

    @staticmethod
    async def finalize(session_id: str, summary: str, start_time: datetime) -> Dict:
        client = get_supabase()
        end_time = datetime.utcnow()
        duration = int((end_time - start_time).total_seconds())
        
        result = await client.table("sessions").update({
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "summary": summary,
            "status": SessionStatus.COMPLETED.value
        }).eq("session_id", session_id).execute()
        
        logger.info(f"Session finalized: {session_id}, duration: {duration}s")
        return result.data[0] if result.data else {}


class EventRepository:
    """CRUD operations for session_events table."""
    
    @staticmethod
    async def log_event(session_id: str, event_type: EventType, event_data: Dict) -> Dict:
        client = get_supabase()
        result = await client.table("session_events").insert({
            "session_id": session_id,
            "event_type": event_type.value,
            "event_data": event_data
        }).execute()
        return result.data[0] if result.data else {}

    @staticmethod
    async def get_conversation_history(session_id: str) -> List[Dict]:
        client = get_supabase()
        result = await client.table("session_events")\
            .select("*")\
            .eq("session_id", session_id)\
            .in_("event_type", [EventType.USER_MESSAGE.value, EventType.AI_RESPONSE.value])\
            .order("sequence_number")\
            .execute()
        
        history = []
        for event in result.data or []:
            role = "user" if event["event_type"] == EventType.USER_MESSAGE.value else "model"
            history.append({"role": role, "content": event["event_data"].get("content", "")})
        return history


# =============================================================================
# TOOL DEFINITIONS & EXECUTOR
# =============================================================================

async def get_weather(city: str, unit: str = "celsius") -> Dict[str, Any]:
    """Simulated weather API call."""
    logger.info(f"Tool: get_weather({city}, {unit})")
    conditions = ["sunny", "cloudy", "partly cloudy", "rainy", "windy"]
    temp_c = random.randint(5, 35)
    temp = temp_c if unit == "celsius" else (temp_c * 9/5) + 32
    return {
        "city": city, "temperature": round(temp, 1), "unit": unit,
        "condition": random.choice(conditions), "humidity": random.randint(30, 90),
        "timestamp": datetime.utcnow().isoformat()
    }


async def search_database(query: str, limit: int = 5) -> Dict[str, Any]:
    """Simulated database search."""
    logger.info(f"Tool: search_database({query}, limit={limit})")
    results = [
        {"id": 1, "title": "Introduction to Machine Learning", "category": "AI"},
        {"id": 2, "title": "Python Best Practices", "category": "Programming"},
        {"id": 3, "title": "Database Optimization", "category": "Database"},
        {"id": 4, "title": "Web Development with FastAPI", "category": "Web"},
        {"id": 5, "title": "Cloud Architecture Patterns", "category": "Cloud"},
    ]
    filtered = [r for r in results if query.lower() in r["title"].lower() or query.lower() in r["category"].lower()]
    return {"query": query, "results": (filtered or results)[:limit], "total": len(filtered or results)}


async def calculate(expression: str) -> Dict[str, Any]:
    """Safe mathematical calculator."""
    logger.info(f"Tool: calculate({expression})")
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return {"expression": expression, "error": "Invalid characters", "result": None}
    try:
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e), "result": None}


async def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """Get current date and time."""
    logger.info(f"Tool: get_current_time({timezone})")
    now = datetime.utcnow()
    return {
        "timezone": timezone, "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A")
    }


# Tool registry
TOOLS = {
    "get_weather": get_weather,
    "search_database": search_database,
    "calculate": calculate,
    "get_current_time": get_current_time
}

# Gemini tool declarations
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
            description="Search database for information",
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
                    "timezone": genai.protos.Schema(type=genai.protos.Type.STRING, description="Timezone")
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

    SUMMARY_PROMPT = """Generate a concise 2-3 sentence summary of this conversation.
Identify main topics, key questions, tools used, and outcomes."""

    INTENT_PATTERNS = {
        "weather": [r"\bweather\b", r"\btemperature\b", r"\brain\b", r"\bsunn?y\b"],
        "math": [r"\bcalculat\w*\b", r"\bcompute\b", r"\b\d+\s*[\+\-\*\/]\s*\d+\b"],
        "technical": [r"\bcode\b", r"\bprogram\w*\b", r"\bapi\b", r"\bdatabase\b"]
    }

    @classmethod
    def get_routed_prompt(cls, message: str) -> str:
        msg_lower = message.lower()
        for intent, patterns in cls.INTENT_PATTERNS.items():
            if any(re.search(p, msg_lower) for p in patterns):
                return {"weather": cls.WEATHER_PROMPT, "math": cls.MATH_PROMPT, 
                        "technical": cls.TECHNICAL_PROMPT}.get(intent, cls.BASE_PROMPT)
        return cls.BASE_PROMPT


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """Gemini API client with streaming and function calling."""
    
    def __init__(self):
        genai.configure(api_key=settings.google_api_key)
        self.chat_sessions: Dict[str, Any] = {}
        logger.info("LLM client initialized")

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

    def get_or_create_chat(self, session_id: str, history: List[Dict] = None, 
                           system_prompt: str = None):
        if session_id not in self.chat_sessions:
            model = self._get_model(system_prompt or PromptManager.BASE_PROMPT)
            formatted = [{"role": m["role"], "parts": [m["content"]]} for m in (history or [])]
            self.chat_sessions[session_id] = model.start_chat(history=formatted)
        return self.chat_sessions[session_id]

    def clear_chat(self, session_id: str):
        self.chat_sessions.pop(session_id, None)

    async def generate_streaming(self, session_id: str, message: str, 
                                  history: List[Dict] = None,
                                  on_tool_call: Callable = None,
                                  on_tool_result: Callable = None):
        """Generate streaming response with tool calling support."""
        system_prompt = PromptManager.BASE_PROMPT
        chat = self.get_or_create_chat(session_id, history, system_prompt)
        
        response = await asyncio.to_thread(chat.send_message, message, stream=True)
        
        full_response = ""
        function_call = None
        
        for chunk in response:
            if hasattr(chunk, 'candidates') and chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                    elif hasattr(part, 'text') and part.text:
                        full_response += part.text
                        yield part.text
        
        # Handle function call
        if function_call:
            tool_name = function_call.name
            tool_args = dict(function_call.args)
            
            if on_tool_call:
                await on_tool_call(tool_name, tool_args)
            
            # Execute tool
            tool_result = await TOOLS[tool_name](**tool_args)
            
            if on_tool_result:
                await on_tool_result(tool_name, tool_result)
            
            # Feed result back to model
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
                            full_response += part.text
                            yield part.text

    async def generate_summary(self, history: List[Dict]) -> str:
        """Generate conversation summary."""
        model = genai.GenerativeModel(
            model_name=settings.llm_model,
            generation_config=GenerationConfig(temperature=0.3, max_output_tokens=300),
            system_instruction=PromptManager.SUMMARY_PROMPT
        )
        
        conversation = "\n".join(
            f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}" for m in history
        )
        
        response = await asyncio.to_thread(
            model.generate_content, f"Summarize:\n\n{conversation}"
        )
        return response.text.strip() if response and response.text else "No summary available."


# Global instances
llm_client = LLMClient()
sessions: Dict[str, SessionState] = {}


# =============================================================================
# POST-SESSION PROCESSING
# =============================================================================

async def process_session_end(session_id: str, start_time: datetime):
    """Background task: generate summary and finalize session."""
    logger.info(f"Post-processing session: {session_id}")
    
    try:
        # Fetch conversation from database
        history = await EventRepository.get_conversation_history(session_id)
        
        if history:
            summary = await llm_client.generate_summary(history)
        else:
            summary = "No conversation recorded."
        
        # Finalize session record
        await SessionRepository.finalize(session_id, summary, start_time)
        logger.info(f"Session {session_id} finalized with summary")
        
    except Exception as e:
        logger.error(f"Post-processing error for {session_id}: {e}")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("Starting RTAI Backend...")
    await init_supabase()
    yield
    await close_supabase()
    logger.info("RTAI Backend shutdown complete")


app = FastAPI(
    title="RTAI Backend",
    description="Real-Time Conversational AI Backend",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "RTAI Backend",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "websocket": "/ws/session/{session_id}"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rtai-backend", "version": "1.0.0"}


@app.websocket("/ws/session/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time conversational sessions.
    
    Protocol:
        Client -> Server: {"type": "user_message", "content": "..."}
        Server -> Client: {"type": "ai_chunk", "content": "..."}
        Server -> Client: {"type": "ai_complete", "content": "..."}
        Server -> Client: {"type": "tool_call", "data": {...}}
        Server -> Client: {"type": "tool_result", "data": {...}}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")
    
    # Get optional user_id from query params
    user_id = websocket.query_params.get("user_id")
    
    # Create session state
    session_state = SessionState(session_id=session_id, user_id=user_id)
    sessions[session_id] = session_state
    
    # Persist session to database
    try:
        await SessionRepository.create(session_id, user_id)
        await EventRepository.log_event(session_id, EventType.SYSTEM_CONNECT, 
                                         {"user_id": user_id, "timestamp": datetime.utcnow().isoformat()})
    except Exception as e:
        logger.error(f"DB error on connect: {e}")
    
    # Send welcome message
    await websocket.send_json(WebSocketMessage(
        type="system",
        content="Connected to ARIA. How can I help you?",
        data={"session_id": session_id}
    ).model_dump())
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            msg_type = data.get("type", "user_message")
            content = data.get("content", "")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            if msg_type != "user_message" or not content:
                continue
            
            logger.info(f"[{session_id}] User: {content[:80]}...")
            
            # Update session state
            session_state.add_message("user", content)
            
            # Log to database
            try:
                await EventRepository.log_event(session_id, EventType.USER_MESSAGE, {"content": content})
            except Exception as e:
                logger.error(f"DB log error: {e}")
            
            # Stream LLM response
            full_response = ""
            
            async def on_tool_call(name: str, args: Dict):
                await websocket.send_json(WebSocketMessage(
                    type="tool_call", data={"tool": name, "arguments": args}
                ).model_dump())
                try:
                    await EventRepository.log_event(session_id, EventType.TOOL_CALL, 
                                                    {"tool": name, "arguments": args})
                except: pass
            
            async def on_tool_result(name: str, result: Dict):
                await websocket.send_json(WebSocketMessage(
                    type="tool_result", data={"tool": name, "result": result}
                ).model_dump())
                try:
                    await EventRepository.log_event(session_id, EventType.TOOL_RESULT,
                                                    {"tool": name, "result": result})
                except: pass
            
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
                
                # Update state
                session_state.add_message("model", full_response)
                
                # Log to database
                try:
                    await EventRepository.log_event(session_id, EventType.AI_RESPONSE, 
                                                    {"content": full_response})
                except: pass
                
                logger.info(f"[{session_id}] AI: {full_response[:80]}...")
                
            except Exception as e:
                logger.error(f"LLM error: {e}")
                await websocket.send_json(WebSocketMessage(type="error", content=str(e)).model_dump())
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Log disconnect
        try:
            await EventRepository.log_event(session_id, EventType.SYSTEM_DISCONNECT,
                                            {"timestamp": datetime.utcnow().isoformat()})
        except: pass
        
        # Trigger post-session processing
        if session_state.message_count > 0:
            asyncio.create_task(process_session_end(session_id, session_state.start_time))
        
        # Cleanup
        sessions.pop(session_id, None)
        llm_client.clear_chat(session_id)
        logger.info(f"Session cleanup complete: {session_id}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
