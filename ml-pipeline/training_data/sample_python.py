"""
Sample Python training data — algorithms, data structures, and utilities.
Add your own .py / .js files alongside this one to expand the vocabulary.
"""

import math
import heapq
import collections
from typing import List, Optional, Dict, Tuple, Any, Iterator


# ── Sorting algorithms ──────────────────────────────────────────────────────

def bubble_sort(arr: List[int]) -> List[int]:
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(arr: List[int], low: int = 0, high: Optional[int] = None) -> List[int]:
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = _partition(arr, low, high)
        quick_sort(arr, low, pivot_idx - 1)
        quick_sort(arr, pivot_idx + 1, high)
    return arr


def _partition(arr: List[int], low: int, high: int) -> int:
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def heap_sort(arr: List[int]) -> List[int]:
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]


# ── Search algorithms ───────────────────────────────────────────────────────

def binary_search(arr: List[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def linear_search(arr: List[Any], target: Any) -> int:
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1


# ── Number theory ───────────────────────────────────────────────────────────

def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fibonacci_sequence(n: int) -> List[int]:
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def sieve_of_eratosthenes(limit: int) -> List[int]:
    is_prime_arr = [True] * (limit + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, limit + 1, i):
                is_prime_arr[j] = False
    return [i for i, flag in enumerate(is_prime_arr) if flag]


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("Factorial undefined for negative integers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# ── Data structures ─────────────────────────────────────────────────────────

class Stack:
    def __init__(self):
        self._items: List[Any] = []

    def push(self, item: Any) -> None:
        self._items.append(item)

    def pop(self) -> Any:
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> Any:
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Stack({self._items!r})"


class Queue:
    def __init__(self):
        self._items: collections.deque = collections.deque()

    def enqueue(self, item: Any) -> None:
        self._items.append(item)

    def dequeue(self) -> Any:
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._items.popleft()

    def front(self) -> Any:
        return self._items[0]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)


class Node:
    def __init__(self, val: Any, next: Optional['Node'] = None):
        self.val = val
        self.next = next


class LinkedList:
    def __init__(self):
        self.head: Optional[Node] = None
        self._size = 0

    def append(self, val: Any) -> None:
        new_node = Node(val)
        if self.head is None:
            self.head = new_node
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            cur.next = new_node
        self._size += 1

    def prepend(self, val: Any) -> None:
        self.head = Node(val, self.head)
        self._size += 1

    def delete(self, val: Any) -> bool:
        if self.head is None:
            return False
        if self.head.val == val:
            self.head = self.head.next
            self._size -= 1
            return True
        cur = self.head
        while cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
                self._size -= 1
                return True
            cur = cur.next
        return False

    def to_list(self) -> List[Any]:
        result = []
        cur = self.head
        while cur:
            result.append(cur.val)
            cur = cur.next
        return result

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return " -> ".join(str(v) for v in self.to_list())


class BSTNode:
    def __init__(self, val: int):
        self.val = val
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None


class BinarySearchTree:
    def __init__(self):
        self.root: Optional[BSTNode] = None

    def insert(self, val: int) -> None:
        self.root = self._insert(self.root, val)

    def _insert(self, node: Optional[BSTNode], val: int) -> BSTNode:
        if node is None:
            return BSTNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        return node

    def search(self, val: int) -> bool:
        return self._search(self.root, val)

    def _search(self, node: Optional[BSTNode], val: int) -> bool:
        if node is None:
            return False
        if val == node.val:
            return True
        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)

    def inorder(self) -> List[int]:
        result: List[int] = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node: Optional[BSTNode], acc: List[int]) -> None:
        if node:
            self._inorder(node.left, acc)
            acc.append(node.val)
            self._inorder(node.right, acc)


# ── Graph algorithms ─────────────────────────────────────────────────────────

class Graph:
    def __init__(self, directed: bool = False):
        self.adj: Dict[int, List[int]] = collections.defaultdict(list)
        self.directed = directed

    def add_edge(self, u: int, v: int) -> None:
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)

    def bfs(self, start: int) -> List[int]:
        visited = set()
        order = []
        queue = collections.deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in self.adj[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        return order

    def dfs(self, start: int) -> List[int]:
        visited = set()
        order: List[int] = []
        self._dfs(start, visited, order)
        return order

    def _dfs(self, node: int, visited: set, order: List[int]) -> None:
        visited.add(node)
        order.append(node)
        for neighbour in self.adj[node]:
            if neighbour not in visited:
                self._dfs(neighbour, visited, order)

    def has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        def dfs_cycle(v: int) -> bool:
            visited.add(v)
            rec_stack.add(v)
            for nb in self.adj[v]:
                if nb not in visited:
                    if dfs_cycle(nb):
                        return True
                elif nb in rec_stack:
                    return True
            rec_stack.discard(v)
            return False

        for node in self.adj:
            if node not in visited:
                if dfs_cycle(node):
                    return True
        return False


def dijkstra(graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, float]:
    dist: Dict[int, float] = collections.defaultdict(lambda: float("inf"))
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in graph.get(u, []):
            nd = dist[u] + weight
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dict(dist)


# ── String utilities ─────────────────────────────────────────────────────────

def is_palindrome(s: str) -> bool:
    s = s.lower().replace(" ", "")
    return s == s[::-1]


def longest_common_subsequence(a: str, b: str) -> str:
    m, n = len(a), len(b)
    dp = [[""] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + a[i - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)
    return dp[m][n]


def count_words(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = collections.defaultdict(int)
    for word in text.lower().split():
        word = word.strip(".,!?;:\"'()")
        if word:
            counts[word] += 1
    return dict(counts)


def reverse_words(sentence: str) -> str:
    return " ".join(sentence.split()[::-1])


def compress_string(s: str) -> str:
    if not s:
        return s
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(s[i - 1] + (str(count) if count > 1 else ""))
            count = 1
    result.append(s[-1] + (str(count) if count > 1 else ""))
    compressed = "".join(result)
    return compressed if len(compressed) < len(s) else s


# ── Dynamic programming ───────────────────────────────────────────────────────

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][capacity]


def coin_change(coins: List[int], amount: int) -> int:
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1


def longest_increasing_subsequence(arr: List[int]) -> int:
    if not arr:
        return 0
    dp = [1] * len(arr)
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


# ── Decorators & generators ──────────────────────────────────────────────────

def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper


@memoize
def fib_memo(n: int) -> int:
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)


def range_infinite(start: int = 0, step: int = 1) -> Iterator[int]:
    n = start
    while True:
        yield n
        n += step


def flatten(nested: List[Any]) -> List[Any]:
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


class Matrix:
    def __init__(self, rows: int, cols: int, fill: float = 0.0):
        self.rows = rows
        self.cols = cols
        self.data = [[fill] * cols for _ in range(rows)]

    def __getitem__(self, pos: Tuple[int, int]) -> float:
        r, c = pos
        return self.data[r][c]

    def __setitem__(self, pos: Tuple[int, int], value: float) -> None:
        r, c = pos
        self.data[r][c] = value

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        assert self.cols == other.rows, "Incompatible matrix dimensions"
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                result[i, j] = sum(self[i, k] * other[k, j] for k in range(self.cols))
        return result

    def transpose(self) -> 'Matrix':
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        return result

    def __repr__(self) -> str:
        return "\n".join(" ".join(f"{v:8.3f}" for v in row) for row in self.data)


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: collections.OrderedDict = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key not in self._cache:
            return -1
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)


class MinHeap:
    def __init__(self):
        self._heap: List[int] = []

    def push(self, val: int) -> None:
        heapq.heappush(self._heap, val)

    def pop(self) -> int:
        return heapq.heappop(self._heap)

    def peek(self) -> int:
        return self._heap[0]

    def __len__(self) -> int:
        return len(self._heap)


def two_sum(nums: List[int], target: int) -> Tuple[int, int]:
    seen: Dict[int, int] = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return seen[complement], i
        seen[num] = i
    return -1, -1


def max_subarray(nums: List[int]) -> int:
    best = current = nums[0]
    for num in nums[1:]:
        current = max(num, current + num)
        best = max(best, current)
    return best


def rotate_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
    return matrix


def group_anagrams(strs: List[str]) -> List[List[str]]:
    groups: Dict[str, List[str]] = collections.defaultdict(list)
    for s in strs:
        key = "".join(sorted(s))
        groups[key].append(s)
    return list(groups.values())


def valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else "#"
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    return not stack
