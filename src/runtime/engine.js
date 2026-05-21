/**
 * src/runtime/engine.js — Autonomous Self-Correction Loop
 *
 * The heart of the agent runtime. Takes a plain-English development task,
 * queries the LLM for a JSON execution plan, executes it, and autonomously
 * retries up to MAX_RETRIES times when the generated code fails.
 *
 * Events emitted (via an EventEmitter):
 *   'log'    — { level, tag, message }  (streamed over SSE)
 *   'done'   — { success, output }
 *   'error'  — { message }
 */

import EventEmitter from 'node:events'
import path         from 'node:path'
import { LLMClient }                                      from '../llm/client.js'
import { fileSystem, dependencyManager, executionUnit }   from './tools.js'

const MAX_RETRIES = 4

// ── System prompt for the planner ─────────────────────────────────────────────

const SYSTEM_CONTEXT = `
You are an expert autonomous software engineer. When given a development task,
you respond ONLY with a JSON object in the exact schema below — no markdown,
no prose, no extra keys.

Schema:
{
  "summary": "<one sentence description of what you are building>",
  "files": [
    {
      "path": "<relative path inside workspace, e.g. server.js>",
      "content": "<full file content as a string>"
    }
  ],
  "install": ["<npm package name>", ...],
  "run": "<shell command to execute the entry point, e.g. node server.js>",
  "verify": "<optional: a secondary shell command to verify, e.g. node --check server.js>"
}

Rules:
- "files" must contain complete, working, runnable code — never placeholder stubs.
- "install" lists only NPM package names (no flags). Empty array if none needed.
- "run" is the single command executed after all files are written.
- "verify" is run first (syntax check, lint, etc.) and is optional.
- All paths are relative to the workspace root.
- For Node.js files use CommonJS (require/module.exports) unless ESM is required.
- For server scripts that listen on a port, use port 5000 and print "LISTENING" when ready.
`.trim()

function buildPlanPrompt(task) {
  return `${SYSTEM_CONTEXT}\n\nDevelopment Task:\n${task}`
}

function buildCorrectionPrompt(task, filePath, fileContent, errorText, attempt) {
  return `${SYSTEM_CONTEXT}

The code you generated previously failed during execution.

Original Task:
${task}

File that caused the error (${filePath}):
\`\`\`
${fileContent}
\`\`\`

Execution Error (attempt ${attempt}/${MAX_RETRIES}):
\`\`\`
${errorText}
\`\`\`

Analyze the error carefully. Rewrite ALL files from scratch to fix the issue.
The corrected code must be complete, syntactically valid, and solve the original task.
Return the full JSON plan again.`
}

// ── Engine ────────────────────────────────────────────────────────────────────

export class Engine extends EventEmitter {
  constructor() {
    super()
    this.llm   = new LLMClient()
    this.busy  = false
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /**
   * Run the full autonomous agent loop for a given task.
   * Emits 'log' events throughout and resolves when the task is complete.
   *
   * @param {string} task — plain English description of what to build
   */
  async run(task) {
    if (this.busy) throw new Error('Engine is already running a task.')
    this.busy = true

    try {
      await fileSystem.ensureWorkspace()
      this.#emit('info', 'AGENT', `Task received: "${task}"`)
      await this.#loop(task)
    } finally {
      this.busy = false
    }
  }

  // ── Internal loop ───────────────────────────────────────────────────────────

  async #loop(task) {
    let plan        = null
    let lastError   = null
    let mainFile    = null
    let mainContent = null

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      // ── 1. Build prompt ────────────────────────────────────────────────────
      let prompt
      if (attempt === 1) {
        this.#emit('info', 'PLANNER', 'Querying LLM for initial execution plan …')
        prompt = buildPlanPrompt(task)
      } else {
        this.#emit('warn', 'AUTONOMOUS RETRY',
          `Sending error log to LLM for self-correction (attempt ${attempt}/${MAX_RETRIES}) …`)
        prompt = buildCorrectionPrompt(task, mainFile, mainContent, lastError, attempt - 1)
      }

      // ── 2. Query LLM ───────────────────────────────────────────────────────
      this.#emit('info', 'LLM REQUEST', `Calling ${this.llm.provider.toUpperCase()} …`)
      let plan
      try {
        plan = await this.llm.generateJSON(prompt)
      } catch (err) {
        this.#emit('error', 'LLM ERROR', err.message)
        this.emit('error', { message: err.message })
        return
      }

      this.#emit('info', 'PLANNER', `Plan summary: ${plan.summary || '(no summary)'}`)
      this.#emit('info', 'PLANNER',
        `Files to write: ${(plan.files || []).length}   ` +
        `Packages to install: ${(plan.install || []).length}   ` +
        `Command: ${plan.run || '(none)'}`)

      // ── 3. Validate plan schema ────────────────────────────────────────────
      if (!Array.isArray(plan.files) || plan.files.length === 0) {
        lastError = 'LLM returned a plan with no files.'
        this.#emit('error', 'PLANNER', lastError)
        continue
      }

      // ── 4. Write files ─────────────────────────────────────────────────────
      this.#emit('info', 'FILE WRITER', `Writing ${plan.files.length} file(s) to workspace …`)
      for (const file of plan.files) {
        if (!file.path || typeof file.content !== 'string') {
          this.#emit('warn', 'FILE WRITER', `Skipping malformed file entry: ${JSON.stringify(file).slice(0, 80)}`)
          continue
        }
        try {
          await fileSystem.write(file.path, file.content)
          this.#emit('ok', 'FILE WRITER', `Written: ${file.path}  (${file.content.length} chars)`)
        } catch (err) {
          this.#emit('error', 'FILE WRITER', `Failed to write ${file.path}: ${err.message}`)
        }
      }

      // Track primary entry point for error-correction context
      const entry = plan.files[0]
      mainFile    = entry.path
      mainContent = entry.content

      // ── 5. Install dependencies ────────────────────────────────────────────
      if (plan.install && plan.install.length > 0) {
        this.#emit('info', 'DEPENDENCY MANAGER',
          `Installing packages: ${plan.install.join(', ')} …`)
        try {
          const out = await dependencyManager.install(plan.install)
          this.#emit('ok', 'DEPENDENCY MANAGER', `Installed OK.\n${out.trim().split('\n').slice(-3).join('\n')}`)
        } catch (err) {
          lastError = `npm install failed: ${err.message}`
          this.#emit('error', 'DEPENDENCY MANAGER', lastError)
          continue
        }
      }

      // ── 6. Optional verify (syntax check) ─────────────────────────────────
      if (plan.verify) {
        this.#emit('info', 'SYNTAX CHECK', `Running: ${plan.verify}`)
        try {
          const { stdout } = await executionUnit.run(plan.verify, { timeout: 15_000 })
          this.#emit('ok', 'SYNTAX CHECK', `Passed.${stdout ? '\n' + stdout.trim() : ''}`)
        } catch (err) {
          lastError = err.message
          this.#emit('error', 'COMPILER ERROR', `Syntax check failed:\n${lastError}`)
          continue
        }
      }

      // ── 7. Execute ─────────────────────────────────────────────────────────
      if (!plan.run) {
        this.#emit('ok', 'AGENT', 'Plan has no run command — task complete (files written only).')
        this.emit('done', { success: true, output: 'Files written successfully.' })
        return
      }

      this.#emit('info', 'TOOL EXECUTION', `Running: ${plan.run}`)
      let result
      try {
        result = await executionUnit.run(plan.run, { timeout: 20_000 })
        const output = result.stdout.trim()
        this.#emit('ok', 'TOOL EXECUTION', `Process exited 0.\n${output}`)
        this.#emit('ok', 'AGENT', `Task completed successfully on attempt ${attempt}.`)
        this.emit('done', { success: true, output })
        return
      } catch (err) {
        lastError = err.message
        this.#emit('error', 'COMPILER ERROR',
          `Execution failed on attempt ${attempt}/${MAX_RETRIES}:\n${lastError}`)

        if (attempt < MAX_RETRIES) {
          this.#emit('warn', 'AUTONOMOUS RETRY',
            `Intercepting error. Feeding back to LLM for self-correction …`)
        }
      }
    }

    // All retries exhausted
    this.#emit('error', 'AGENT',
      `Task failed after ${MAX_RETRIES} attempts. Last error:\n${lastError}`)
    this.emit('done', { success: false, output: lastError })
  }

  // ── Logging helper ──────────────────────────────────────────────────────────

  #emit(level, tag, message) {
    const entry = {
      level,
      tag,
      message,
      ts: new Date().toISOString(),
    }
    this.emit('log', entry)
    // Also write to process stdout for debugging
    const prefix = { info: '●', ok: '✓', warn: '⚠', error: '✗' }[level] || '·'
    console.log(`${prefix} [${tag}] ${message}`)
  }
}
