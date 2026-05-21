/**
 * src/llm/client.js — Agnostic LLM Gateway
 *
 * Priority order:
 *   1. GEMINI_API_KEY present → Google Gemini via @google/genai SDK
 *   2. No key              → Local Ollama at http://localhost:11434
 *
 * All responses are strictly validated as JSON before returning.
 */

import { GoogleGenAI } from '@google/genai'

const GEMINI_MODEL = 'gemini-2.5-flash'
const OLLAMA_URL   = process.env.OLLAMA_URL || 'http://localhost:11434/api/generate'
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'llama3.2'
const MAX_TOKENS   = 8192
const MAX_JSON_RETRIES = 3

// ── Provider detection ────────────────────────────────────────────────────────

function detectProvider() {
  if (process.env.GEMINI_API_KEY) return 'gemini'
  return 'ollama'
}

// ── JSON strict extractor ─────────────────────────────────────────────────────

/**
 * Extract and parse the first valid JSON object from a model response.
 * Handles markdown code fences like ```json ... ```.
 * Throws if no valid JSON found.
 */
function extractJSON(raw) {
  if (!raw || typeof raw !== 'string') {
    throw new Error('Model returned empty response.')
  }

  // Strip markdown code fences
  let cleaned = raw
    .replace(/^```(?:json)?\s*/im, '')
    .replace(/```\s*$/im, '')
    .trim()

  // Attempt direct parse
  try {
    return JSON.parse(cleaned)
  } catch (_) { /* fall through */ }

  // Try to extract the outermost { … } block
  const start = cleaned.indexOf('{')
  const end   = cleaned.lastIndexOf('}')
  if (start !== -1 && end !== -1 && end > start) {
    try {
      return JSON.parse(cleaned.slice(start, end + 1))
    } catch (_) { /* fall through */ }
  }

  throw new Error(
    `Model did not return valid JSON.\n--- Raw output (first 500 chars) ---\n${raw.slice(0, 500)}`
  )
}

// ── Gemini provider ───────────────────────────────────────────────────────────

async function callGemini(prompt) {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

  const response = await ai.models.generateContent({
    model:    GEMINI_MODEL,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: {
      responseMimeType: 'application/json',
      maxOutputTokens:  MAX_TOKENS,
      temperature:      0.2,
    },
  })

  const text = response.text
  if (!text) throw new Error('Gemini returned an empty response.')
  return text
}

// ── Ollama provider ───────────────────────────────────────────────────────────

async function callOllama(prompt) {
  const body = JSON.stringify({
    model:  OLLAMA_MODEL,
    prompt: `You are a code generation assistant. Always respond with valid JSON only — no markdown, no prose.\n\n${prompt}`,
    stream: false,
    options: { temperature: 0.2, num_predict: MAX_TOKENS },
  })

  let res
  try {
    res = await fetch(OLLAMA_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    })
  } catch (err) {
    throw new Error(
      `Cannot reach Ollama at ${OLLAMA_URL}.\n` +
      `Make sure Ollama is running: ollama serve\n` +
      `Original error: ${err.message}`
    )
  }

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Ollama HTTP ${res.status}: ${text.slice(0, 300)}`)
  }

  const data = await res.json()
  if (!data.response) throw new Error('Ollama returned no response field.')
  return data.response
}

// ── Public LLMClient class ────────────────────────────────────────────────────

export class LLMClient {
  constructor() {
    this.provider = detectProvider()
    console.log(`[LLMClient] Provider: ${this.provider.toUpperCase()}`)
    if (this.provider === 'gemini') {
      console.log(`[LLMClient] Model: ${GEMINI_MODEL}`)
    } else {
      console.log(`[LLMClient] Model: ${OLLAMA_MODEL}  Endpoint: ${OLLAMA_URL}`)
    }
  }

  /**
   * Send a prompt to the active LLM provider and return a validated JSON object.
   * Retries up to MAX_JSON_RETRIES times if the model produces invalid JSON.
   *
   * @param {string} prompt
   * @returns {Promise<object>}
   */
  async generateJSON(prompt) {
    for (let attempt = 1; attempt <= MAX_JSON_RETRIES; attempt++) {
      let raw
      try {
        raw = this.provider === 'gemini'
          ? await callGemini(prompt)
          : await callOllama(prompt)
      } catch (err) {
        if (attempt === MAX_JSON_RETRIES) throw err
        console.warn(`[LLMClient] Provider error (attempt ${attempt}): ${err.message}`)
        await sleep(1000 * attempt)
        continue
      }

      try {
        return extractJSON(raw)
      } catch (jsonErr) {
        if (attempt === MAX_JSON_RETRIES) {
          throw new Error(`JSON validation failed after ${MAX_JSON_RETRIES} attempts:\n${jsonErr.message}`)
        }
        console.warn(`[LLMClient] Invalid JSON on attempt ${attempt}, retrying with correction …`)
        // Prepend correction instruction on retry
        prompt =
          `Your previous response was not valid JSON. Parse error: ${jsonErr.message}\n\n` +
          `Original request:\n${prompt}\n\n` +
          `Respond ONLY with a valid JSON object. No markdown, no explanations.`
      }
    }
  }

  /**
   * Convenience wrapper used by the engine for task planning.
   * Returns the raw text (for logging), then the parsed JSON.
   */
  async plan(taskPrompt) {
    return this.generateJSON(taskPrompt)
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}
