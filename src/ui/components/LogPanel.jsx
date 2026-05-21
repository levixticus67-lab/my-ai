import React, { useEffect, useRef } from 'react'

// ── Level → colour mapping ────────────────────────────────────────────────────

const LEVEL_STYLE = {
  info:  { prefix: '●', cls: 'text-[#58a6ff]',  tagCls: 'text-[#58a6ff]'  },
  ok:    { prefix: '✓', cls: 'text-[#3fb950]',  tagCls: 'text-[#3fb950]'  },
  warn:  { prefix: '⚠', cls: 'text-[#d29922]',  tagCls: 'text-[#d29922]'  },
  error: { prefix: '✗', cls: 'text-[#f85149]',  tagCls: 'text-[#f85149]'  },
}

const TAG_ACCENT = {
  'TOOL EXECUTION':      'bg-[#1f3a5f] text-[#58a6ff]',
  'COMPILER ERROR':      'bg-[#3d1a1a] text-[#f85149]',
  'AUTONOMOUS RETRY':    'bg-[#3d2e00] text-[#d29922]',
  'LLM REQUEST':         'bg-[#1e2a1e] text-[#39d353]',
  'FILE WRITER':         'bg-[#1a1f2e] text-[#bc8cff]',
  'DEPENDENCY MANAGER':  'bg-[#1a1f2e] text-[#bc8cff]',
  'SYNTAX CHECK':        'bg-[#1f2a1f] text-[#39d353]',
  'PLANNER':             'bg-[#1a2a3a] text-[#58a6ff]',
  'AGENT':               'bg-[#0d2010] text-[#3fb950]',
  'SERVER':              'bg-[#1f1f1f] text-[#8b949e]',
}

// ── Formatters ────────────────────────────────────────────────────────────────

function formatTime(iso) {
  try {
    return new Date(iso).toISOString().slice(11, 23)   // HH:mm:ss.mmm
  } catch {
    return '??:??:??.???'
  }
}

function tagClass(tag) {
  return TAG_ACCENT[tag] || 'bg-[#161b22] text-[#8b949e]'
}

// ── Log entry ─────────────────────────────────────────────────────────────────

function LogEntry({ entry }) {
  const { level = 'info', tag, message, ts } = entry
  const { prefix, cls, tagCls } = LEVEL_STYLE[level] || LEVEL_STYLE.info

  // Multi-line messages — indent continuation lines
  const lines = (message || '').split('\n')

  return (
    <div className="flex gap-2 py-[3px] items-start group hover:bg-[#161b22] px-2 rounded">
      {/* Timestamp */}
      <span className="shrink-0 text-[10px] text-[#3d444d] pt-px mt-[1px] select-none w-[80px]">
        {formatTime(ts)}
      </span>

      {/* Level prefix */}
      <span className={`shrink-0 text-xs font-bold ${cls} w-3 mt-px select-none`}>
        {prefix}
      </span>

      {/* Tag badge */}
      <span className={`shrink-0 text-[10px] font-bold px-1.5 py-px rounded ${tagClass(tag)} select-none whitespace-nowrap`}>
        {tag}
      </span>

      {/* Message */}
      <span className="flex-1 text-xs text-[#e6edf3] break-words">
        {lines.map((line, i) => (
          <span key={i} className={i > 0 ? 'block text-[#8b949e]' : ''}>
            {i > 0 ? '    ' + line : line}
          </span>
        ))}
      </span>
    </div>
  )
}

// ── Idle / running placeholders ───────────────────────────────────────────────

function IdlePrompt() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8 select-none">
      <div className="text-4xl mb-4 text-[#30363d]">⬡</div>
      <p className="text-sm text-[#8b949e] mb-2">Agent is idle.</p>
      <p className="text-xs text-[#3d444d]">
        Submit a task above — the engine loop will stream its operations here in real time.
      </p>
    </div>
  )
}

function RunningSpinner() {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-[#d29922] px-2 py-1">
      <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#d29922] animate-bounce [animation-delay:0ms]" />
      <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#d29922] animate-bounce [animation-delay:150ms]" />
      <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#d29922] animate-bounce [animation-delay:300ms]" />
      <span className="ml-1">Engine running…</span>
    </span>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function LogPanel({ logs, status }) {
  const bottomRef = useRef(null)

  // Auto-scroll to bottom on new logs
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  if (logs.length === 0 && status === 'idle') {
    return (
      <div className="flex-1 overflow-y-auto bg-[#0d1117]">
        <IdlePrompt />
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-[#0d1117]">
      <div className="flex-1 overflow-y-auto py-2 px-1">
        {logs.map(entry => (
          <LogEntry key={entry.id} entry={entry} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Running indicator pinned at bottom */}
      {status === 'running' && (
        <div className="shrink-0 border-t border-[#30363d] px-2 py-1 bg-[#161b22]">
          <RunningSpinner />
        </div>
      )}
    </div>
  )
}
