/**
 * src/runtime/tools.js — System Execution Handlers
 *
 * All operations are sandboxed inside ./workspace to prevent the engine
 * from touching files outside the project directory.
 *
 * Exports:
 *   fileSystem        — write / read / append / exists / list
 *   dependencyManager — npm install inside workspace
 *   executionUnit     — run scripts/commands inside workspace
 */

import { exec }     from 'node:child_process'
import { promisify } from 'node:util'
import fs            from 'node:fs/promises'
import path          from 'node:path'

const execAsync = promisify(exec)

// Absolute path to the sandbox workspace
export const WORKSPACE_DIR = path.resolve('./workspace')

// ── Safety guard ──────────────────────────────────────────────────────────────

/**
 * Resolve a relative path to an absolute path inside WORKSPACE_DIR.
 * Throws if the resolved path escapes the sandbox (path traversal prevention).
 */
function safeResolve(relativePath) {
  const resolved = path.resolve(WORKSPACE_DIR, relativePath)
  if (!resolved.startsWith(WORKSPACE_DIR + path.sep) && resolved !== WORKSPACE_DIR) {
    throw new Error(
      `[SECURITY] Path "${relativePath}" resolved outside workspace: "${resolved}"`
    )
  }
  return resolved
}

// ── File System ───────────────────────────────────────────────────────────────

export const fileSystem = {
  /**
   * Write content to a file, creating parent directories as needed.
   * Overwrites any existing file.
   */
  async write(relativePath, content) {
    const abs = safeResolve(relativePath)
    await fs.mkdir(path.dirname(abs), { recursive: true })
    await fs.writeFile(abs, content, 'utf-8')
    return abs
  },

  /**
   * Read a file's content as a string.
   */
  async read(relativePath) {
    const abs = safeResolve(relativePath)
    return fs.readFile(abs, 'utf-8')
  },

  /**
   * Append content to an existing file (creates if absent).
   */
  async append(relativePath, content) {
    const abs = safeResolve(relativePath)
    await fs.mkdir(path.dirname(abs), { recursive: true })
    await fs.appendFile(abs, content, 'utf-8')
    return abs
  },

  /**
   * Check whether a file or directory exists in the workspace.
   */
  async exists(relativePath) {
    const abs = safeResolve(relativePath)
    return fs.access(abs).then(() => true).catch(() => false)
  },

  /**
   * Recursively list all files in the workspace as relative paths.
   * Returns an array of objects: { path, size, mtime }.
   */
  async list(dir = '.') {
    const absDir = safeResolve(dir)
    const results = []
    await walk(absDir, WORKSPACE_DIR, results)
    return results
  },

  /**
   * Delete a file from the workspace.
   */
  async remove(relativePath) {
    const abs = safeResolve(relativePath)
    await fs.rm(abs, { force: true })
  },

  /**
   * Ensure the workspace directory exists (idempotent).
   */
  async ensureWorkspace() {
    await fs.mkdir(WORKSPACE_DIR, { recursive: true })
  },
}

async function walk(absDir, root, results) {
  let entries
  try {
    entries = await fs.readdir(absDir, { withFileTypes: true })
  } catch {
    return
  }
  for (const entry of entries) {
    // Skip node_modules and hidden dirs for cleanliness
    if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue
    const absPath = path.join(absDir, entry.name)
    if (entry.isDirectory()) {
      await walk(absPath, root, results)
    } else {
      const rel = path.relative(root, absPath)
      const stat = await fs.stat(absPath).catch(() => null)
      results.push({
        path:  rel,
        size:  stat ? stat.size : 0,
        mtime: stat ? stat.mtime.toISOString() : null,
      })
    }
  }
}

// ── Dependency Manager ────────────────────────────────────────────────────────

export const dependencyManager = {
  /**
   * Run `npm install <packages>` inside the workspace.
   * Initialises package.json first if it does not exist.
   *
   * @param {string[]} packages — e.g. ['express', 'cors']
   * @returns {Promise<string>} stdout from npm
   */
  async install(packages) {
    await fileSystem.ensureWorkspace()

    // Ensure package.json exists
    const pkgPath = path.join(WORKSPACE_DIR, 'package.json')
    try {
      await fs.access(pkgPath)
    } catch {
      await fs.writeFile(
        pkgPath,
        JSON.stringify({ name: 'workspace', version: '1.0.0', type: 'commonjs' }, null, 2),
        'utf-8'
      )
    }

    const pkgList = packages.join(' ')
    const cmd     = `npm install ${pkgList} --save --prefer-offline 2>&1`

    const { stdout, stderr } = await execAsync(cmd, {
      cwd:     WORKSPACE_DIR,
      timeout: 120_000,   // 2 minutes
    })
    return stdout + (stderr ? `\nSTDERR:\n${stderr}` : '')
  },

  /**
   * Run `npm install` (no arguments) to restore an existing package.json.
   */
  async restore() {
    await fileSystem.ensureWorkspace()
    const { stdout } = await execAsync('npm install --prefer-offline 2>&1', {
      cwd:     WORKSPACE_DIR,
      timeout: 120_000,
    })
    return stdout
  },
}

// ── Execution Unit ────────────────────────────────────────────────────────────

export const executionUnit = {
  /**
   * Execute a shell command inside the workspace.
   *
   * Resolves with stdout on exit code 0.
   * Rejects with the exact stderr string on non-zero exit.
   *
   * @param {string}  command  — e.g. 'node server.js' or 'npm run build'
   * @param {object}  opts
   * @param {number}  opts.timeout  — ms before SIGTERM (default 30 000)
   * @param {object}  opts.env      — extra env vars merged with process.env
   * @returns {Promise<{ stdout: string, code: number }>}
   */
  run(command, { timeout = 30_000, env = {} } = {}) {
    return new Promise((resolve, reject) => {
      const child = exec(command, {
        cwd:     WORKSPACE_DIR,
        timeout,
        env:     { ...process.env, ...env },
      })

      let stdout = ''
      let stderr = ''

      child.stdout?.on('data', chunk => { stdout += chunk })
      child.stderr?.on('data', chunk => { stderr += chunk })

      child.on('close', (code, signal) => {
        if (signal) {
          return reject(new Error(
            `Process killed by signal ${signal} after ${timeout}ms.\n${stderr}`
          ))
        }
        if (code !== 0) {
          return reject(new Error(
            `Exit code ${code}.\n${stderr || stdout}`
          ))
        }
        resolve({ stdout, code })
      })

      child.on('error', err => reject(err))
    })
  },

  /**
   * Syntax-check a Node.js file without executing it.
   * Uses `node --check`.
   *
   * @param {string} relativePath
   * @returns {Promise<void>}  Resolves silently on success, rejects with error text.
   */
  async syntaxCheck(relativePath) {
    const abs = safeResolve(relativePath)
    try {
      await execAsync(`node --check "${abs}"`, { cwd: WORKSPACE_DIR, timeout: 10_000 })
    } catch (err) {
      throw new Error(`Syntax error in ${relativePath}:\n${err.stderr || err.message}`)
    }
  },
}
