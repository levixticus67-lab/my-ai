/**
 * src/api/server.js — Express Daemon with SSE streaming
 *
 * Routes:
 *   GET  /api/health          → { status, provider, busy }
 *   POST /api/run             → Submit a task (starts the engine)
 *   GET  /api/stream          → SSE event stream of live engine logs
 *   GET  /api/workspace-tree  → Recursive list of workspace files
 *   GET  /api/workspace-file  → Read a single workspace file (?path=…)
 *   DELETE /api/workspace     → Clear the workspace directory
 */

import express    from 'express'
import cors       from 'cors'
import path       from 'node:path'
import fs         from 'node:fs/promises'
import { Engine } from '../runtime/engine.js'
import { fileSystem, WORKSPACE_DIR } from '../runtime/tools.js'

const PORT = parseInt(process.env.SERVER_PORT || '4000', 10)
const app  = express()

// ── Middleware ─────────────────────────────────────────────────────────────────

app.use(cors())
app.use(express.json({ limit: '2mb' }))

// ── Global engine instance (singleton) ────────────────────────────────────────

const engine = new Engine()

// ── SSE client registry ────────────────────────────────────────────────────────

/** @type {Set<import('express').Response>} */
const sseClients = new Set()

/** Broadcast a structured message to all connected SSE clients */
function broadcast(type, payload) {
  const data = JSON.stringify({ type, payload, ts: new Date().toISOString() })
  for (const res of sseClients) {
    try {
      res.write(`data: ${data}\n\n`)
    } catch {
      sseClients.delete(res)
    }
  }
}

// Wire engine events → SSE broadcast
engine.on('log', entry => {
  broadcast('log', entry)
})

engine.on('done', result => {
  broadcast('done', result)
})

engine.on('error', err => {
  broadcast('error', err)
})

// ── Routes ────────────────────────────────────────────────────────────────────

/**
 * GET /api/health
 * Health check — always fast, no side effects.
 */
app.get('/api/health', (_req, res) => {
  res.json({
    status:   'ok',
    provider: engine.llm.provider,
    busy:     engine.busy,
    uptime:   Math.round(process.uptime()),
  })
})

/**
 * GET /api/stream
 * Server-Sent Events — clients receive all engine log events in real time.
 *
 * Event format:
 *   data: { type: "log"|"done"|"error"|"ping", payload: {...}, ts: "ISO string" }
 */
app.get('/api/stream', (req, res) => {
  res.setHeader('Content-Type',  'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection',    'keep-alive')
  res.setHeader('X-Accel-Buffering', 'no')   // disable nginx buffering
  res.flushHeaders()

  // Register client
  sseClients.add(res)

  // Send immediate confirmation
  const welcome = JSON.stringify({
    type:    'connected',
    payload: { message: 'SSE stream connected', clients: sseClients.size },
    ts:      new Date().toISOString(),
  })
  res.write(`data: ${welcome}\n\n`)

  // Keep-alive ping every 25 s
  const pingInterval = setInterval(() => {
    const ping = JSON.stringify({ type: 'ping', ts: new Date().toISOString() })
    try { res.write(`data: ${ping}\n\n`) } catch { /* client gone */ }
  }, 25_000)

  // Cleanup on disconnect
  req.on('close', () => {
    sseClients.delete(res)
    clearInterval(pingInterval)
  })
})

/**
 * POST /api/run
 * Body: { task: string }
 * Starts the autonomous engine loop. Returns immediately with 202 Accepted.
 * All progress is streamed over /api/stream.
 */
app.post('/api/run', async (req, res) => {
  const { task } = req.body

  if (!task || typeof task !== 'string' || task.trim().length < 5) {
    return res.status(400).json({ error: 'task must be a non-empty string (min 5 chars).' })
  }

  if (engine.busy) {
    return res.status(409).json({ error: 'Engine is already running a task. Wait for it to finish.' })
  }

  // Respond immediately — client watches SSE for progress
  res.status(202).json({ accepted: true, task: task.trim() })

  // Run asynchronously
  broadcast('log', {
    level:   'info',
    tag:     'SERVER',
    message: `Task accepted: "${task.trim()}"`,
    ts:      new Date().toISOString(),
  })

  engine.run(task.trim()).catch(err => {
    broadcast('error', { message: err.message })
  })
})

/**
 * GET /api/workspace-tree
 * Returns the list of files in the workspace directory.
 */
app.get('/api/workspace-tree', async (_req, res) => {
  try {
    await fileSystem.ensureWorkspace()
    const files = await fileSystem.list()
    res.json({ files, root: WORKSPACE_DIR })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

/**
 * GET /api/workspace-file?path=<relpath>
 * Read a single file from the workspace.
 */
app.get('/api/workspace-file', async (req, res) => {
  const { path: relPath } = req.query
  if (!relPath) return res.status(400).json({ error: 'path query param required.' })

  try {
    const content = await fileSystem.read(relPath)
    res.json({ path: relPath, content })
  } catch (err) {
    res.status(404).json({ error: `File not found: ${relPath}` })
  }
})

/**
 * DELETE /api/workspace
 * Wipe the workspace directory and recreate it empty.
 */
app.delete('/api/workspace', async (_req, res) => {
  if (engine.busy) {
    return res.status(409).json({ error: 'Cannot clear workspace while engine is running.' })
  }
  try {
    await fs.rm(WORKSPACE_DIR, { recursive: true, force: true })
    await fileSystem.ensureWorkspace()
    broadcast('log', {
      level: 'warn', tag: 'SERVER',
      message: 'Workspace cleared.', ts: new Date().toISOString(),
    })
    res.json({ cleared: true })
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// ── 404 fallback ──────────────────────────────────────────────────────────────

app.use((req, res) => {
  res.status(404).json({ error: `No route: ${req.method} ${req.path}` })
})

// ── Start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`\n┌─────────────────────────────────────────────────`)
  console.log(`│  Agent Runtime Server`)
  console.log(`│  API   → http://localhost:${PORT}/api`)
  console.log(`│  SSE   → http://localhost:${PORT}/api/stream`)
  console.log(`│  LLM   → ${engine.llm.provider.toUpperCase()}`)
  console.log(`└─────────────────────────────────────────────────\n`)
})
