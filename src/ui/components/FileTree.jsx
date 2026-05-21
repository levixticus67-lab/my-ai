import React, { useState, useEffect, useCallback } from 'react'

const API = ''

// ── Icon helpers ──────────────────────────────────────────────────────────────

function fileIcon(filePath) {
  const ext = filePath.split('.').pop()?.toLowerCase()
  const map = {
    js:   '󰌞',   jsx: '⚛',  ts: '󰛦',   tsx: '⚛',
    json: '{}',  md:  '󰍔',  css: '󰌜',  html: '󰌝',
    sh:   '$',   env: '🔑',  lock: '🔒', txt: '󰈚',
  }
  return map[ext] || '󰈔'
}

function extColor(filePath) {
  const ext = filePath.split('.').pop()?.toLowerCase()
  const map = {
    js: 'text-[#d29922]', jsx: 'text-[#58a6ff]',
    ts: 'text-[#58a6ff]', tsx: 'text-[#58a6ff]',
    json: 'text-[#3fb950]', md: 'text-[#8b949e]',
    css: 'text-[#bc8cff]', html: 'text-[#f85149]',
    sh: 'text-[#39d353]', lock: 'text-[#3d444d]',
  }
  return map[ext] || 'text-[#e6edf3]'
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes}B`
  return `${(bytes / 1024).toFixed(1)}K`
}

// ── Tree builder: flat list → nested structure ────────────────────────────────

function buildTree(files) {
  const root = {}
  for (const file of files) {
    const parts = file.path.replace(/\\/g, '/').split('/')
    let node = root
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i]
      if (i === parts.length - 1) {
        // Leaf file
        node[part] = { __file: true, ...file }
      } else {
        if (!node[part]) node[part] = {}
        node = node[part]
      }
    }
  }
  return root
}

// ── File viewer modal ─────────────────────────────────────────────────────────

function FileViewer({ filePath, onClose }) {
  const [content, setContent] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch(`${API}/api/workspace-file?path=${encodeURIComponent(filePath)}`)
      .then(r => r.json())
      .then(d => {
        if (d.error) throw new Error(d.error)
        setContent(d.content)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [filePath])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={onClose}>
      <div
        className="bg-[#161b22] border border-[#30363d] rounded-lg w-[80vw] max-h-[80vh] flex flex-col overflow-hidden shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-[#30363d]">
          <span className="text-xs font-mono text-[#58a6ff]">{filePath}</span>
          <button onClick={onClose} className="text-[#8b949e] hover:text-[#e6edf3] text-sm">✕</button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {loading && <span className="text-[#8b949e] text-xs">Loading…</span>}
          {error   && <span className="text-[#f85149] text-xs">{error}</span>}
          {content !== null && (
            <pre className="text-xs text-[#e6edf3] leading-relaxed whitespace-pre-wrap break-words">
              {content}
            </pre>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Recursive tree renderer ───────────────────────────────────────────────────

function TreeNode({ name, node, depth = 0, onFileClick }) {
  const [open, setOpen] = useState(true)
  const indent = depth * 12

  if (node.__file) {
    // Leaf file
    return (
      <button
        onClick={() => onFileClick(node.path)}
        className="flex items-center gap-1.5 w-full text-left px-2 py-[2px] hover:bg-[#21262d] rounded group"
        style={{ paddingLeft: `${indent + 8}px` }}
      >
        <span className={`text-xs ${extColor(name)}`}>{fileIcon(name)}</span>
        <span className={`text-xs flex-1 truncate ${extColor(name)}`}>{name}</span>
        <span className="text-[10px] text-[#3d444d] group-hover:text-[#8b949e] shrink-0">
          {formatSize(node.size || 0)}
        </span>
      </button>
    )
  }

  // Directory
  const children = Object.entries(node).sort(([aName, aNode], [bName, bNode]) => {
    // Dirs before files
    const aDir = !aNode.__file
    const bDir = !bNode.__file
    if (aDir !== bDir) return bDir - aDir
    return aName.localeCompare(bName)
  })

  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 w-full text-left px-2 py-[2px] hover:bg-[#21262d] rounded"
        style={{ paddingLeft: `${indent + 8}px` }}
      >
        <span className="text-[10px] text-[#8b949e] w-2">{open ? '▾' : '▸'}</span>
        <span className="text-[10px] text-[#8b949e]">📁</span>
        <span className="text-xs text-[#8b949e] font-semibold">{name}/</span>
      </button>
      {open && children.map(([childName, childNode]) => (
        <TreeNode
          key={childName}
          name={childName}
          node={childNode}
          depth={depth + 1}
          onFileClick={onFileClick}
        />
      ))}
    </div>
  )
}

// ── Main FileTree component ───────────────────────────────────────────────────

export default function FileTree({ version }) {
  const [files,    setFiles]    = useState([])
  const [loading,  setLoading]  = useState(false)
  const [viewing,  setViewing]  = useState(null)
  const [error,    setError]    = useState(null)

  const refresh = useCallback(() => {
    setLoading(true)
    setError(null)
    fetch(`${API}/api/workspace-tree`)
      .then(r => r.json())
      .then(d => {
        if (d.error) throw new Error(d.error)
        setFiles(d.files || [])
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  // Refresh when version bumps
  useEffect(() => { refresh() }, [version, refresh])

  // Auto-refresh every 5 s while viewing
  useEffect(() => {
    const id = setInterval(refresh, 5000)
    return () => clearInterval(id)
  }, [refresh])

  const tree = buildTree(files)
  const roots = Object.keys(tree)

  return (
    <div className="flex-1 overflow-y-auto bg-[#0d1117] py-2">

      {/* Status bar */}
      <div className="flex items-center gap-2 px-3 pb-2 border-b border-[#30363d] mb-2">
        <span className="text-[10px] text-[#8b949e]">
          {files.length} file{files.length !== 1 ? 's' : ''}
        </span>
        {loading && <span className="text-[10px] text-[#d29922] animate-pulse ml-auto">syncing…</span>}
        <button
          onClick={refresh}
          className="ml-auto text-[10px] text-[#58a6ff] hover:text-[#79c0ff]"
        >
          ↺
        </button>
      </div>

      {error && (
        <p className="text-xs text-[#f85149] px-3">{error}</p>
      )}

      {files.length === 0 && !loading && !error && (
        <div className="flex flex-col items-center justify-center h-32 text-center px-4 select-none">
          <p className="text-xs text-[#3d444d]">Workspace is empty.</p>
          <p className="text-[10px] text-[#3d444d] mt-1">Files appear here once the engine writes them.</p>
        </div>
      )}

      {roots.map(name => (
        <TreeNode
          key={name}
          name={name}
          node={tree[name]}
          depth={0}
          onFileClick={setViewing}
        />
      ))}

      {viewing && (
        <FileViewer filePath={viewing} onClose={() => setViewing(null)} />
      )}
    </div>
  )
}
