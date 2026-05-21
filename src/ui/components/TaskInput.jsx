import React, { useState, useRef } from 'react'

const EXAMPLES = [
  'Create a local Express server file with an active port listening for JSON payloads, verify its syntax, and install the cors package automatically.',
  'Build a Node.js script that reads a JSON file, filters items where price > 50, and writes the result to output.json.',
  'Create a simple HTTP server that responds with the current UTC time as JSON on GET /time.',
  'Write a Node.js script that fetches the GitHub API for the top 5 trending repositories and prints their names and star counts.',
]

export default function TaskInput({ onSubmit, disabled }) {
  const [value,    setValue]    = useState('')
  const [expanded, setExpanded] = useState(false)
  const textareaRef = useRef(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    const task = value.trim()
    if (!task || disabled) return
    onSubmit(task)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit(e)
    }
    if (e.key === 'Escape') {
      setExpanded(false)
    }
  }

  const useExample = (ex) => {
    setValue(ex)
    setExpanded(false)
    textareaRef.current?.focus()
  }

  return (
    <div className="shrink-0 border-b border-[#30363d] bg-[#161b22]">
      <form onSubmit={handleSubmit} className="px-4 py-3">
        <div className="flex gap-3 items-start">

          {/* Prompt symbol */}
          <span className="text-[#3fb950] font-bold pt-2 text-sm select-none">$</span>

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={value}
            onChange={e => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setExpanded(true)}
            placeholder="Describe the program you want the agent to build…  (Ctrl+Enter to run)"
            rows={expanded ? 3 : 1}
            disabled={disabled}
            className="flex-1 bg-transparent text-[#e6edf3] text-sm resize-none outline-none placeholder-[#8b949e] font-mono leading-relaxed transition-all duration-150 disabled:opacity-50"
          />

          {/* Submit button */}
          <button
            type="submit"
            disabled={disabled || !value.trim()}
            className={`
              shrink-0 px-4 py-1.5 rounded text-xs font-bold tracking-widest uppercase transition-all duration-150
              ${disabled
                ? 'bg-[#21262d] text-[#8b949e] cursor-not-allowed'
                : 'bg-[#238636] hover:bg-[#2ea043] text-white cursor-pointer'
              }
            `}
          >
            {disabled ? 'RUNNING…' : 'RUN'}
          </button>
        </div>

        {/* Examples bar */}
        {expanded && !disabled && (
          <div className="mt-2 ml-5">
            <p className="text-xs text-[#8b949e] mb-1.5">Quick examples:</p>
            <div className="flex flex-col gap-1">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => useExample(ex)}
                  className="text-left text-xs text-[#58a6ff] hover:text-[#79c0ff] hover:underline truncate"
                >
                  › {ex}
                </button>
              ))}
            </div>
          </div>
        )}
      </form>
    </div>
  )
}
