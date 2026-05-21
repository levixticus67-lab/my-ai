# Agent Runtime — Local Autonomous Multi-Agent Engine

A fully self-contained, event-driven autonomous agent that takes a plain-English programming task, plans execution using an LLM, writes real code files, installs dependencies, runs them, and **self-corrects autonomously** up to 4 times when the generated code fails — all streamed live to a terminal-style dashboard.

No LangChain. No wrappers. Native Node.js core loop.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT RUNTIME                        │
│                                                         │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │  LLM Client  │────▶│   Engine     │                  │
│  │  (client.js) │◀────│  (engine.js) │                  │
│  └──────────────┘     └──────┬───────┘                  │
│   Gemini / Ollama            │                          │
│                       ┌──────▼───────┐                  │
│                       │    Tools     │                  │
│                       │  (tools.js)  │                  │
│                       │  fileSystem  │                  │
│                       │  depManager  │                  │
│                       │  execUnit    │                  │
│                       └──────────────┘                  │
│                              │                          │
│              ┌───────────────▼────────────────┐         │
│              │        ./workspace/             │         │
│              │   (all generated code lives     │         │
│              │    here — fully sandboxed)       │         │
│              └────────────────────────────────┘         │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │    Express API Server  (server.js)               │   │
│  │    POST /api/run        → start task             │   │
│  │    GET  /api/stream     → SSE log stream         │   │
│  │    GET  /api/workspace-tree → file list          │   │
│  │    DELETE /api/workspace   → clear sandbox       │   │
│  └──────────────────────────────────────────────────┘   │
│                         ▲                               │
│  ┌──────────────────────┴───────────────────────────┐   │
│  │    React Dashboard  (src/ui/)                    │   │
│  │    Left panel  → live SSE engine logs            │   │
│  │    Right panel → workspace file tree             │   │
│  │    Input bar   → submit tasks                    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
cd agent-runtime
npm install
```

### 2. Configure your LLM

**Option A — Google Gemini (recommended, cloud):**
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```
Get a free key at https://aistudio.google.com/apikey

**Option B — Local Ollama (fully offline, no key needed):**
```bash
ollama serve           # start Ollama daemon
ollama pull llama3.2   # or any model you prefer
# No .env needed — engine auto-detects absence of GEMINI_API_KEY
```

### 3. Start everything

```bash
npm run dev
```

This starts two processes concurrently:
- **API server** on `http://localhost:4000`
- **React dashboard** on `http://localhost:3000`

Open `http://localhost:3000` in your browser.

---

## How It Works — The Autonomous Loop

```
User submits task
       │
       ▼
  LLM generates JSON plan
  { files[], install[], run, verify }
       │
       ▼
  Engine writes files → ./workspace/
       │
       ▼
  npm install packages (if any)
       │
       ▼
  Syntax check (verify command)
       │
    PASS? ──NO──▶ Build correction prompt with error text
       │                    │
       │         Re-query LLM (up to 4 times)
       │                    │
       ▼                    ▼
  Execute (run command)   Retry loop
       │
    EXIT 0? ──NO──▶ Intercept stderr
       │            Build correction prompt
       │            Re-query LLM (up to 4 times)
       ▼
  DONE ✓
```

Every step emits a named log event (`[TOOL EXECUTION]`, `[COMPILER ERROR]`, `[AUTONOMOUS RETRY]`, etc.) that streams over SSE to the dashboard.

---

## Test Case: The Built-in Demo

Submit this task in the dashboard input:

> **"Create a local Express server file with an active port listening for JSON payloads, verify its syntax, and install the cors package automatically."**

The engine will:
1. Query the LLM → receive a JSON plan with `server.js`, `install: ["express","cors"]`, `run: "node --check server.js"`
2. Write `workspace/server.js`
3. Run `npm install express cors` inside `workspace/`
4. Run `node --check server.js` (syntax verify)
5. Execute `node server.js` — server binds, prints `LISTENING`, exits 0
6. Dashboard shows all steps streamed live; right panel shows `server.js` appear in the file tree

If the LLM writes broken code on attempt 1, the engine automatically feeds the exact `stderr` back with a structured correction prompt and retries.

---

## File Structure

```
agent-runtime/
├── index.html                   Vite HTML entry
├── package.json                 Dependencies + npm scripts
├── vite.config.js               Vite: port 3000, proxy /api → :4000
├── tailwind.config.js
├── postcss.config.js
├── .env.example                 Copy to .env and add GEMINI_API_KEY
│
├── src/
│   ├── llm/
│   │   └── client.js            LLM gateway (Gemini / Ollama, JSON-strict)
│   │
│   ├── runtime/
│   │   ├── tools.js             fileSystem · dependencyManager · executionUnit
│   │   └── engine.js            Autonomous self-correction loop (EventEmitter)
│   │
│   ├── api/
│   │   └── server.js            Express + SSE daemon (port 4000)
│   │
│   └── ui/
│       ├── index.css            Tailwind entry
│       ├── main.jsx             React entry
│       ├── App.jsx              Root: SSE connection, layout, state
│       └── components/
│           ├── TaskInput.jsx    Task submit bar + quick examples
│           ├── LogPanel.jsx     Left: colour-coded live log stream
│           └── FileTree.jsx     Right: workspace directory explorer
│
└── workspace/                   SANDBOX — all generated code lives here
    └── (created at runtime)
```

---

## Module Deep-Dives

### `src/llm/client.js` — Agnostic LLM Gateway

- Detects `GEMINI_API_KEY` → uses `@google/genai` SDK with `gemini-2.5-flash`
- No key → falls back to `http://localhost:11434/api/generate` (Ollama)
- **Strict JSON enforcement**: extracts JSON from markdown fences, retries up to 3× with a self-correction prompt if the model returns invalid JSON
- Configurable via `OLLAMA_URL` and `OLLAMA_MODEL` env vars

### `src/runtime/tools.js` — System Handlers

Three modules, all sandboxed to `./workspace`:

| Module | Functions |
|--------|-----------|
| `fileSystem` | `write`, `read`, `append`, `exists`, `list`, `remove`, `ensureWorkspace` |
| `dependencyManager` | `install(packages[])`, `restore()` — runs real `npm install` |
| `executionUnit` | `run(command, {timeout, env})` — returns `{stdout, code}` or throws with exact stderr |

Path traversal is blocked at the `safeResolve()` layer — no `../../` escapes.

### `src/runtime/engine.js` — The Core Loop

A stateful `EventEmitter` class. Runs up to `MAX_RETRIES = 4` times per task:

1. Builds a system-prompt + task → structured JSON plan
2. Writes all files, installs packages, runs verify + run commands
3. On any failure: captures exact `stderr`, constructs a correction prompt with the original task + broken file content + error text, re-queries the LLM
4. Emits `log`, `done`, `error` events consumed by the SSE server

### `src/api/server.js` — The Daemon

| Route | Method | Description |
|-------|--------|-------------|
| `/api/health` | GET | Status, provider, uptime, busy flag |
| `/api/stream` | GET | SSE — all engine events in real time |
| `/api/run` | POST | Submit `{ task }` — 202 Accepted, progress via SSE |
| `/api/workspace-tree` | GET | Recursive file list with sizes |
| `/api/workspace-file` | GET | Read a file `?path=relative/path` |
| `/api/workspace` | DELETE | Clear the sandbox |

---

## SSE Event Schema

Every event over `/api/stream` is:
```json
{
  "type":    "log | done | error | connected | ping",
  "payload": { "level": "info|ok|warn|error", "tag": "AGENT", "message": "…" },
  "ts":      "2025-01-01T12:00:00.000Z"
}
```

Log tag examples and their colours in the dashboard:

| Tag | Colour | Meaning |
|-----|--------|---------|
| `TOOL EXECUTION` | Blue | Running a shell command |
| `COMPILER ERROR` | Red | Non-zero exit / syntax failure |
| `AUTONOMOUS RETRY` | Yellow | Sending error back to LLM |
| `LLM REQUEST` | Green | Calling the model |
| `FILE WRITER` | Purple | Writing files to workspace |
| `DEPENDENCY MANAGER` | Purple | npm install running |
| `AGENT` | Green | Top-level loop status |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key. If absent, Ollama is used. |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model to use |
| `SERVER_PORT` | `4000` | Express server port |

---

## Stack

- **Runtime**: Node.js 20+ (ES Modules)
- **LLM**: Google Gemini (`@google/genai`) or local Ollama
- **API**: Express 4, Server-Sent Events
- **Frontend**: React 18, Vite 5, Tailwind CSS 3
- **Tools**: native `node:child_process`, `node:fs/promises` — zero wrapper libraries
