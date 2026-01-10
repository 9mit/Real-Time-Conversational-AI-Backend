# RTAI - Real-Time Conversational AI Backend

A high-performance, asynchronous Python backend implementing real-time bi-directional WebSocket communication with LLM streaming, function calling, async database persistence, and automated post-session processing.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/9mit/Real-Time-Conversational-AI-Backend.git
cd Real-Time-Conversational-AI-Backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
# Create .env file and add your credentials:
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_KEY=your-service-role-key
# GOOGLE_API_KEY=your-gemini-api-key

# Run the server
python main.py

# Open tests/test_client.html in your browser to test
```

## Features

- **Real-time WebSocket Streaming**: Token-by-token LLM response delivery
- **Function/Tool Calling**: 4 integrated tools with LLM result integration
- **Multi-Step Routing**: Dynamic system prompts based on user intent
- **State Management**: Full conversation history across multiple turns
- **Async Supabase Persistence**: Sessions + event logging
- **Post-Session Automation**: Background summary generation on disconnect

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│                    (WebSocket Client)                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Real-time Messages
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   WebSocket     │  │    Session      │  │      LLM        │  │
│  │   Handler       │──│    Manager      │──│    Service      │  │
│  │ /ws/session/id  │  │  (State Mgmt)   │  │   (Streaming)   │  │
│  └─────────────────┘  └─────────────────┘  └────────┬────────┘  │
│                                                      │           │
│                                             ┌────────▼────────┐  │
│                                             │  Tool Executor  │  │
│                                             │ (Function Call) │  │
│                                             └─────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Async Operations
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Persistence Layer                            │
│  ┌─────────────────┐                    ┌─────────────────────┐  │
│  │  Sessions Table │                    │  Session Events     │  │
│  │  (Metadata)     │◄──────────────────►│  (Event Log)        │  │
│  └─────────────────┘                    └─────────────────────┘  │
│                         Supabase PostgreSQL                      │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
RTAI/
├── main.py             # Complete FastAPI backend (single-file)
├── schema.sql          # Supabase database schema
├── requirements.txt    # Python dependencies
├── .env                # Environment configuration
├── .gitignore          # Git exclusions
├── README.md           # Documentation
└── tests/
    ├── test_client.html    # Browser WebSocket test UI
    └── test_websocket.py   # CLI test client
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| Server | Uvicorn |
| LLM | Google Gemini 2.5 Flash |
| Database | Supabase PostgreSQL |
| Validation | Pydantic |

## Setup & Installation

### Prerequisites
- Python 3.10+
- Google Gemini API key
- Supabase account

### 1. Create Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Supabase
1. Create a project at [supabase.com](https://supabase.com)
2. Run the SQL from `schema.sql` in the SQL Editor
3. Get your Project URL and service_role key from Settings → API

### 4. Configure Environment
Create a `.env` file:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
GOOGLE_API_KEY=your-gemini-api-key
```

### 5. Run the Server
```bash
python main.py
```

## Testing

### Browser Test Client
Open `tests/test_client.html` in a browser.

### CLI Test Client
```bash
python tests/test_websocket.py
```

### Health Check
```bash
curl http://localhost:5000/health
```

## WebSocket Protocol

### Endpoint
```
ws://localhost:5000/ws/session/{session_id}
```

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `user_message` | Client→Server | User input |
| `ai_chunk` | Server→Client | Streaming token |
| `ai_complete` | Server→Client | Full response |
| `tool_call` | Server→Client | Tool invocation |
| `tool_result` | Server→Client | Tool output |

### Example
```json
{"type": "user_message", "content": "What's the weather in Tokyo?"}
{"type": "tool_call", "data": {"tool": "get_weather", "arguments": {"city": "Tokyo"}}}
{"type": "ai_chunk", "content": "The weather in Tokyo..."}
{"type": "ai_complete", "content": "The weather in Tokyo is sunny with 25°C."}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `get_weather` | Fetch weather data for a city |
| `search_database` | Search for information |
| `calculate` | Perform math calculations |
| `get_current_time` | Get current date/time |

## Database Schema

```sql
-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration_seconds INTEGER,
    summary TEXT,
    status VARCHAR(50)
);

-- Events table
CREATE TABLE session_events (
    id UUID PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id),
    event_type VARCHAR(50),
    event_data JSONB,
    timestamp TIMESTAMPTZ,
    sequence_number SERIAL
);
```

## License

MIT License
