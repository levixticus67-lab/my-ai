/**
 * Sample JavaScript training data — algorithms, utilities, and patterns.
 * Add your own .py / .js files to expand the vocabulary and improve generation.
 */

"use strict";

// ── Sorting ──────────────────────────────────────────────────────────────────

function bubbleSort(arr) {
  const a = [...arr];
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a.length - i - 1; j++) {
      if (a[j] > a[j + 1]) [a[j], a[j + 1]] = [a[j + 1], a[j]];
    }
  }
  return a;
}

function mergeSort(arr) {
  if (arr.length <= 1) return arr;
  const mid = Math.floor(arr.length / 2);
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));
  return merge(left, right);
}

function merge(left, right) {
  const result = [];
  let i = 0, j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] <= right[j]) result.push(left[i++]);
    else result.push(right[j++]);
  }
  return result.concat(left.slice(i)).concat(right.slice(j));
}

function quickSort(arr, lo = 0, hi = arr.length - 1) {
  if (lo < hi) {
    const p = partition(arr, lo, hi);
    quickSort(arr, lo, p - 1);
    quickSort(arr, p + 1, hi);
  }
  return arr;
}

function partition(arr, lo, hi) {
  const pivot = arr[hi];
  let i = lo - 1;
  for (let j = lo; j < hi; j++) {
    if (arr[j] <= pivot) { i++; [arr[i], arr[j]] = [arr[j], arr[i]]; }
  }
  [arr[i + 1], arr[hi]] = [arr[hi], arr[i + 1]];
  return i + 1;
}

// ── Search ───────────────────────────────────────────────────────────────────

function binarySearch(arr, target) {
  let lo = 0, hi = arr.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] === target) return mid;
    else if (arr[mid] < target) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

// ── Fibonacci & number theory ────────────────────────────────────────────────

function fibonacci(n) {
  if (n <= 1) return n;
  let a = 0, b = 1;
  for (let i = 2; i <= n; i++) [a, b] = [b, a + b];
  return b;
}

function* fibonacciGenerator() {
  let a = 0, b = 1;
  while (true) {
    yield a;
    [a, b] = [b, a + b];
  }
}

function isPrime(n) {
  if (n < 2) return false;
  if (n === 2) return true;
  if (n % 2 === 0) return false;
  for (let i = 3; i <= Math.sqrt(n); i += 2) {
    if (n % i === 0) return false;
  }
  return true;
}

function sieve(limit) {
  const flags = new Array(limit + 1).fill(true);
  flags[0] = flags[1] = false;
  for (let i = 2; i * i <= limit; i++) {
    if (flags[i]) {
      for (let j = i * i; j <= limit; j += i) flags[j] = false;
    }
  }
  return flags.reduce((acc, f, i) => { if (f) acc.push(i); return acc; }, []);
}

function gcd(a, b) { return b === 0 ? a : gcd(b, a % b); }
function lcm(a, b) { return Math.abs(a * b) / gcd(a, b); }
function factorial(n) { return n <= 1 ? 1 : n * factorial(n - 1); }

// ── Data structures ──────────────────────────────────────────────────────────

class Stack {
  #items = [];
  push(item)  { this.#items.push(item); }
  pop()       { if (this.isEmpty()) throw new Error("Stack underflow"); return this.#items.pop(); }
  peek()      { if (this.isEmpty()) throw new Error("Empty stack"); return this.#items.at(-1); }
  isEmpty()   { return this.#items.length === 0; }
  get size()  { return this.#items.length; }
  toString()  { return `Stack[${this.#items.join(", ")}]`; }
}

class Queue {
  #items = [];
  enqueue(item) { this.#items.push(item); }
  dequeue()     { if (this.isEmpty()) throw new Error("Queue underflow"); return this.#items.shift(); }
  front()       { return this.#items[0]; }
  isEmpty()     { return this.#items.length === 0; }
  get size()    { return this.#items.length; }
}

class ListNode {
  constructor(val, next = null) { this.val = val; this.next = next; }
}

class LinkedList {
  constructor() { this.head = null; this.length = 0; }

  append(val) {
    const node = new ListNode(val);
    if (!this.head) { this.head = node; }
    else {
      let cur = this.head;
      while (cur.next) cur = cur.next;
      cur.next = node;
    }
    this.length++;
  }

  delete(val) {
    if (!this.head) return false;
    if (this.head.val === val) { this.head = this.head.next; this.length--; return true; }
    let cur = this.head;
    while (cur.next) {
      if (cur.next.val === val) { cur.next = cur.next.next; this.length--; return true; }
      cur = cur.next;
    }
    return false;
  }

  toArray() {
    const arr = []; let cur = this.head;
    while (cur) { arr.push(cur.val); cur = cur.next; }
    return arr;
  }

  reverse() {
    let prev = null, cur = this.head;
    while (cur) { const next = cur.next; cur.next = prev; prev = cur; cur = next; }
    this.head = prev;
    return this;
  }
}

class HashMap {
  constructor(capacity = 16) {
    this.capacity = capacity;
    this.size = 0;
    this.buckets = new Array(capacity).fill(null).map(() => []);
  }

  #hash(key) {
    let h = 0;
    for (const ch of String(key)) h = (h * 31 + ch.charCodeAt(0)) % this.capacity;
    return h;
  }

  set(key, value) {
    const idx = this.#hash(key);
    const bucket = this.buckets[idx];
    const entry = bucket.find(([k]) => k === key);
    if (entry) { entry[1] = value; } else { bucket.push([key, value]); this.size++; }
  }

  get(key) {
    const entry = this.buckets[this.#hash(key)].find(([k]) => k === key);
    return entry ? entry[1] : undefined;
  }

  has(key) { return !!this.buckets[this.#hash(key)].find(([k]) => k === key); }

  delete(key) {
    const idx = this.#hash(key);
    const bucket = this.buckets[idx];
    const i = bucket.findIndex(([k]) => k === key);
    if (i === -1) return false;
    bucket.splice(i, 1); this.size--; return true;
  }

  keys()   { return this.buckets.flatMap(b => b.map(([k]) => k)); }
  values() { return this.buckets.flatMap(b => b.map(([, v]) => v)); }
}

class MinHeap {
  #heap = [];
  push(val)   { this.#heap.push(val); this.#bubbleUp(); }
  pop()       { this.#swap(0, this.#heap.length - 1); const v = this.#heap.pop(); this.#sinkDown(); return v; }
  peek()      { return this.#heap[0]; }
  get size()  { return this.#heap.length; }

  #bubbleUp() {
    let i = this.#heap.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.#heap[parent] <= this.#heap[i]) break;
      this.#swap(parent, i); i = parent;
    }
  }

  #sinkDown() {
    let i = 0;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < this.#heap.length && this.#heap[l] < this.#heap[smallest]) smallest = l;
      if (r < this.#heap.length && this.#heap[r] < this.#heap[smallest]) smallest = r;
      if (smallest === i) break;
      this.#swap(i, smallest); i = smallest;
    }
  }

  #swap(i, j) { [this.#heap[i], this.#heap[j]] = [this.#heap[j], this.#heap[i]]; }
}

// ── Graph ────────────────────────────────────────────────────────────────────

class Graph {
  constructor(directed = false) {
    this.adj = new Map();
    this.directed = directed;
  }

  addEdge(u, v, w = 1) {
    if (!this.adj.has(u)) this.adj.set(u, []);
    if (!this.adj.has(v)) this.adj.set(v, []);
    this.adj.get(u).push({ node: v, weight: w });
    if (!this.directed) this.adj.get(v).push({ node: u, weight: w });
  }

  bfs(start) {
    const visited = new Set([start]);
    const queue = [start], order = [];
    while (queue.length) {
      const node = queue.shift();
      order.push(node);
      for (const { node: nb } of (this.adj.get(node) || [])) {
        if (!visited.has(nb)) { visited.add(nb); queue.push(nb); }
      }
    }
    return order;
  }

  dfs(start) {
    const visited = new Set(), order = [];
    const dfsHelper = (v) => {
      visited.add(v); order.push(v);
      for (const { node: nb } of (this.adj.get(v) || [])) {
        if (!visited.has(nb)) dfsHelper(nb);
      }
    };
    dfsHelper(start);
    return order;
  }

  dijkstra(start) {
    const dist = new Map();
    this.adj.forEach((_, k) => dist.set(k, Infinity));
    dist.set(start, 0);
    const pq = new MinHeap();
    pq.push([0, start]);
    while (pq.size) {
      const [d, u] = pq.pop();
      if (d > dist.get(u)) continue;
      for (const { node: v, weight: w } of (this.adj.get(u) || [])) {
        const nd = dist.get(u) + w;
        if (nd < (dist.get(v) ?? Infinity)) { dist.set(v, nd); pq.push([nd, v]); }
      }
    }
    return dist;
  }
}

// ── Dynamic programming ───────────────────────────────────────────────────────

function knapsack(weights, values, capacity) {
  const n = weights.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(capacity + 1).fill(0));
  for (let i = 1; i <= n; i++) {
    for (let w = 0; w <= capacity; w++) {
      dp[i][w] = dp[i - 1][w];
      if (weights[i - 1] <= w)
        dp[i][w] = Math.max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]);
    }
  }
  return dp[n][capacity];
}

function coinChange(coins, amount) {
  const dp = new Array(amount + 1).fill(Infinity);
  dp[0] = 0;
  for (const coin of coins)
    for (let i = coin; i <= amount; i++)
      dp[i] = Math.min(dp[i], dp[i - coin] + 1);
  return dp[amount] === Infinity ? -1 : dp[amount];
}

function longestCommonSubsequence(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
  return dp[m][n];
}

function maxSubarray(nums) {
  let best = nums[0], cur = nums[0];
  for (let i = 1; i < nums.length; i++) {
    cur = Math.max(nums[i], cur + nums[i]);
    best = Math.max(best, cur);
  }
  return best;
}

// ── String utilities ──────────────────────────────────────────────────────────

function isPalindrome(s) {
  const clean = s.toLowerCase().replace(/[^a-z0-9]/g, "");
  return clean === clean.split("").reverse().join("");
}

function countWords(text) {
  const counts = {};
  for (const word of text.toLowerCase().match(/\b\w+\b/g) || [])
    counts[word] = (counts[word] || 0) + 1;
  return counts;
}

function groupAnagrams(strs) {
  const map = new Map();
  for (const s of strs) {
    const key = s.split("").sort().join("");
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(s);
  }
  return [...map.values()];
}

function longestPalindrome(s) {
  let start = 0, maxLen = 1;
  const expand = (l, r) => {
    while (l >= 0 && r < s.length && s[l] === s[r]) { l--; r++; }
    if (r - l - 1 > maxLen) { maxLen = r - l - 1; start = l + 1; }
  };
  for (let i = 0; i < s.length; i++) { expand(i, i); expand(i, i + 1); }
  return s.slice(start, start + maxLen);
}

function validParentheses(s) {
  const stack = [], map = { ")": "(", "}": "{", "]": "[" };
  for (const ch of s) {
    if (map[ch]) { if (stack.pop() !== map[ch]) return false; }
    else stack.push(ch);
  }
  return stack.length === 0;
}

// ── Functional utilities ──────────────────────────────────────────────────────

const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);
const compose = (...fns) => x => fns.reduceRight((v, f) => f(v), x);
const curry = fn => {
  const arity = fn.length;
  return function curried(...args) {
    return args.length >= arity ? fn(...args) : (...more) => curried(...args, ...more);
  };
};
const memoize = fn => {
  const cache = new Map();
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) return cache.get(key);
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
};
const debounce = (fn, delay) => {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
};
const throttle = (fn, limit) => {
  let last = 0;
  return (...args) => { const now = Date.now(); if (now - last >= limit) { last = now; fn(...args); } };
};
const chunk = (arr, size) => {
  const result = [];
  for (let i = 0; i < arr.length; i += size) result.push(arr.slice(i, i + size));
  return result;
};
const flatten = arr => arr.reduce((a, v) => a.concat(Array.isArray(v) ? flatten(v) : v), []);
const unique = arr => [...new Set(arr)];
const zip = (...arrays) => arrays[0].map((_, i) => arrays.map(a => a[i]));

// ── Promise utilities ────────────────────────────────────────────────────────

async function retry(fn, attempts = 3, delay = 500) {
  for (let i = 0; i < attempts; i++) {
    try { return await fn(); }
    catch (err) {
      if (i === attempts - 1) throw err;
      await new Promise(r => setTimeout(r, delay * 2 ** i));
    }
  }
}

function timeout(promise, ms) {
  const t = new Promise((_, reject) => setTimeout(() => reject(new Error("Timeout")), ms));
  return Promise.race([promise, t]);
}

async function* asyncRange(start, end, step = 1) {
  for (let i = start; i < end; i += step) {
    yield i;
    await new Promise(r => setTimeout(r, 0));
  }
}

// ── Event emitter ────────────────────────────────────────────────────────────

class EventEmitter {
  #listeners = new Map();

  on(event, fn) {
    if (!this.#listeners.has(event)) this.#listeners.set(event, []);
    this.#listeners.get(event).push(fn);
    return this;
  }

  off(event, fn) {
    if (!this.#listeners.has(event)) return this;
    this.#listeners.set(event, this.#listeners.get(event).filter(l => l !== fn));
    return this;
  }

  once(event, fn) {
    const wrapper = (...args) => { fn(...args); this.off(event, wrapper); };
    return this.on(event, wrapper);
  }

  emit(event, ...args) {
    for (const fn of (this.#listeners.get(event) || [])) fn(...args);
    return this;
  }
}

// ── Observable (minimal RxJS-like) ───────────────────────────────────────────

class Observable {
  constructor(subscriber) { this._subscriber = subscriber; }

  subscribe(observer) {
    return this._subscriber(
      typeof observer === "function" ? { next: observer } : observer
    );
  }

  map(fn) {
    return new Observable(obs => this.subscribe({ next: v => obs.next(fn(v)) }));
  }

  filter(fn) {
    return new Observable(obs => this.subscribe({ next: v => { if (fn(v)) obs.next(v); } }));
  }

  static of(...values) {
    return new Observable(obs => { values.forEach(v => obs.next(v)); });
  }

  static fromPromise(p) {
    return new Observable(obs => p.then(v => obs.next(v)));
  }
}

// ── LRU Cache ────────────────────────────────────────────────────────────────

class LRUCache {
  #capacity; #map = new Map();

  constructor(capacity) { this.#capacity = capacity; }

  get(key) {
    if (!this.#map.has(key)) return -1;
    const val = this.#map.get(key);
    this.#map.delete(key);
    this.#map.set(key, val);
    return val;
  }

  put(key, value) {
    if (this.#map.has(key)) this.#map.delete(key);
    else if (this.#map.size >= this.#capacity) this.#map.delete(this.#map.keys().next().value);
    this.#map.set(key, value);
  }
}

module.exports = {
  bubbleSort, mergeSort, quickSort, binarySearch,
  fibonacci, fibonacciGenerator, isPrime, sieve, gcd, lcm, factorial,
  Stack, Queue, LinkedList, HashMap, MinHeap, Graph,
  knapsack, coinChange, longestCommonSubsequence, maxSubarray,
  isPalindrome, countWords, groupAnagrams, longestPalindrome, validParentheses,
  pipe, compose, curry, memoize, debounce, throttle, chunk, flatten, unique, zip,
  retry, timeout, asyncRange, EventEmitter, Observable, LRUCache,
};
