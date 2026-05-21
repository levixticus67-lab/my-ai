import React, { useState, useEffect, useRef, useCallback } from 'react'
import TaskInput   from './components/TaskInput'
import LogPanel    from './components/LogPanel'
import FileTree    from './components/FileTree'

const API = ''   // empty = same origin (proxied by Vite → port 4000)

export default function App() {
  const [logs,        setLogs]        = useState([])
  const [status,      setStatus]      = useState('idle')   // idle | running | done | error
  const [connected,   setConnected]   = useState(false)
  const [provider,    setProvider]    = useState('…')
  const [treeVersion, setTreeVersion] = useState(0)        // bump to refresh file tree

  const eventSourceRef = useRef(null)

  // ── SSE connection ──────────────────────────────────────────────────────────

  const connectSSE = useCallback(() => {
    if (eventSourceRef.current) eventSourceRef.current.close()

    const es = new EventSource(`${API}/api/stream`)
    eventSourceRef.current = es

    es.onopen = () => setConnected(true)

    es.onmessage = (event) => {
      let msg
      try { msg = JSON.parse(event.data) } catch { return }

      if (msg.type === 'connected') {
        setConnected(true)
        return
      }

      if (msg.type === 'ping') return

      if (msg.type === 'log') {
        setLogs(prev => [...prev, { ...msg.payload, id: Date.now() + Math.random() }])
        return
      }

      if (msg.type === 'done') {
        setStatus(msg.payload.success ? 'done' : 'error')
        setTreeVersion(v => v + 1)
        setLogs(prev => [
          ...prev,
          {
            id:      Date.now(),
            level:   msg.payload.success ? 'ok' : 'error',
            tag:     'AGENT',
            message: msg.payload.success
              ? `✓ Task complete.\n${msg.payload.output || ''}`
              : `✗ Task failed.\n${msg.payload.output || ''}`,
            ts: new Date().toISOString(),
          }
        ])
        return
      }

      if (msg.type === 'error') {
        setStatus('error')
        setLogs(prev => [
          ...prev,
          { id: Date.now(), level: 'error', tag: 'SERVER', message: msg.payload.message, ts: new Date().toISOString() }
        ])
      }
    }

    es.onerror = () => {
      setConnected(false)
      setTimeout(connectSSE, 3000)   // reconnect
    }
  }, [])

  useEffect(() => {
    connectSSE()
    return () => eventSourceRef.current?.close()
  }, [connectSSE])

  // ── Fetch provider info ─────────────────────────────────────────────────────

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then(r => r.json())
      .then(d => setProvider(d.provider?.toUpperCase() || '?'))
      .catch(() => {})

    const id = setInterval(() => {
      fetch(`${API}/api/health`)
        .then(r => r.json())
        .then(d => setProvider(d.provider?.toUpperCase() || '?'))
        .catch(() => {})
    }, 10_000)
    return () => clearInterval(id)
  }, [])

  // ── Submit task ─────────────────────────────────────────────────────────────

  const handleSubmit = async (task) => {
    setLogs([])
    setStatus('running')
    setTreeVersion(v => v + 1)

    try {
      const res = await fetch(`${API}/api/run`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ task }),
      })
      if (!res.ok) {
        const err = await res.json()
        setStatus('error')
        setLogs([{ id: Date.now(), level: 'error', tag: 'CLIENT', message: err.error, ts: new Date().toISOString() }])
      }
    } catch (err) {
      setStatus('error')
      setLogs([{ id: Date.now(), level: 'error', tag: 'CLIENT', message: `Cannot reach server: ${err.message}`, ts: new Date().toISOString() }])
    }
  }

  const handleClear = async () => {
    try {
      await fetch(`${API}/api/workspace`, { method: 'DELETE' })
      setLogs([])
      setStatus('idle')
      setTreeVersion(v => v + 1)
    } catch { /* ignore */ }
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen bg-[#0d1117] text-[#e6edf3] overflow-hidden">

      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="flex items-center gap-3 px-5 py-3 border-b border-[#30363d] bg-[#161b22] shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[#3fb950] font-bold text-sm tracking-widest">AGENT</span>
          <span className="text-[#30363d]">|</span>
          <span className="text-[#8b949e] text-xs">RUNTIME ENGINE</span>
        </div>

        <div className="flex items-center gap-4 ml-auto text-xs text-[#8b949e]">
          {/* Connection status */}
          <span className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-[#3fb950]' : 'bg-[#f85149]'} animate-pulse`} />
            {connected ? 'SSE LIVE' : 'RECONNECTING'}
          </span>

          {/* Engine status */}
          <span className="flex items-center gap-1.5">
            <StatusBadge status={status} />
          </span>

          {/* Provider */}
          <span className="text-[#58a6ff]">LLM: {provider}</span>

          {/* Clear */}
          <button
            onClick={handleClear}
            disabled={status === 'running'}
            className="px-2 py-0.5 rounded border border-[#30363d] hover:border-[#8b949e] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            CLEAR WORKSPACE
          </button>
        </div>
      </header>

      {/* ── Task Input ──────────────────────────────────────────── */}
      <TaskInput onSubmit={handleSubmit} disabled={status === 'running'} />

      {/* ── Split Panels ────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left: Log stream */}
        <div className="flex flex-col flex-1 border-r border-[#30363d] overflow-hidden">
          <PanelHeader label="ENGINE LOGS" badge={logs.length} color="cyan" />
          <LogPanel logs={logs} status={status} />
        </div>

        {/* Right: File tree */}
        <div className="flex flex-col w-80 overflow-hidden shrink-0">
          <PanelHeader label="WORKSPACE" color="purple" />
          <FileTree version={treeVersion} />
        </div>

      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function PanelHeader({ label, badge, color }) {
  const colors = {
    cyan:   'text-[#39d353]',
    purple: 'text-[#bc8cff]',
  }
  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-[#30363d] bg-[#161b22] shrink-0">
      <span className={`text-xs font-semibold tracking-widest ${colors[color] || ''}`}>
        {label}
      </span>
      {badge !== undefined && (
        <span className="ml-auto text-xs text-[#8b949e]">{badge} events</span>
      )}
    </div>
  )
}

function StatusBadge({ status }) {
  const map = {
    idle:    { label: 'IDLE',    cls: 'text-[#8b949e]' },
    running: { label: 'RUNNING', cls: 'text-[#d29922] animate-pulse' },
    done:    { label: 'DONE',    cls: 'text-[#3fb950]' },
    error:   { label: 'FAILED',  cls: 'text-[#f85149]' },
  }
  const { label, cls } = map[status] || map.idle
  return <span className={`font-bold ${cls}`}>{label}</span>
}
