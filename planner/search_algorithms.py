"""
SmartPrep AI - Search Algorithms for Study Planner
Implements BFS, DFS, UCS, A*, Hill Climbing to generate optimized study schedules.
"""

import time
from collections import deque
import heapq
import random

random.seed(42)

# ─────────────────────────────────────────────
# STUDY GRAPH DEFINITION
# ─────────────────────────────────────────────

def build_study_graph(topics: list, weak_topics: list = None):
    """
    Build a weighted directed graph of topics.
    Edge weight = estimated study hours (lower for weak topics = higher priority).
    """
    if weak_topics is None:
        weak_topics = []
    
    graph = {}
    for i, topic in enumerate(topics):
        neighbors = []
        # Connect each topic to 1-3 subsequent topics
        for j in range(i + 1, min(i + 3, len(topics))):
            # Weight: weak topics get lower weight (higher priority in UCS/A*)
            weight = 1 if topics[j] in weak_topics else 2
            neighbors.append((topics[j], weight))
        graph[topic] = neighbors
    
    return graph


def heuristic(topic: str, goal: str, weak_topics: list):
    """A* heuristic: weak topics have lower heuristic (prioritized)."""
    if topic in weak_topics:
        return 0.5
    return 1.0


# ─────────────────────────────────────────────
# BFS - Breadth First Search
# ─────────────────────────────────────────────

def bfs_study_plan(graph: dict, start: str, goal: str = None):
    """
    BFS: Explores level by level.
    Returns: ordered study sequence, path, nodes explored, time taken.
    """
    t0 = time.perf_counter()
    visited = []
    queue = deque([start])
    seen = {start}
    path = []

    while queue:
        node = queue.popleft()
        visited.append(node)
        path.append(node)
        if goal and node == goal:
            break
        for neighbor, _ in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    elapsed = round((time.perf_counter() - t0) * 1000, 3)
    return {
        "algorithm": "BFS",
        "path": path,
        "nodes_explored": len(visited),
        "time_ms": elapsed,
        "description": "Level-by-level exploration; guarantees shortest path in unweighted graphs."
    }


# ─────────────────────────────────────────────
# DFS - Depth First Search
# ─────────────────────────────────────────────

def dfs_study_plan(graph: dict, start: str, goal: str = None):
    """
    DFS: Explores as deep as possible before backtracking.
    """
    t0 = time.perf_counter()
    visited = []
    stack = [start]
    seen = set()
    path = []

    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)
        path.append(node)
        if goal and node == goal:
            break
        for neighbor, _ in reversed(graph.get(node, [])):
            if neighbor not in seen:
                stack.append(neighbor)

    elapsed = round((time.perf_counter() - t0) * 1000, 3)
    return {
        "algorithm": "DFS",
        "path": path,
        "nodes_explored": len(visited),
        "time_ms": elapsed,
        "description": "Deep-first exploration; memory efficient but may not find optimal path."
    }


# ─────────────────────────────────────────────
# UCS - Uniform Cost Search
# ─────────────────────────────────────────────

def ucs_study_plan(graph: dict, start: str, goal: str = None):
    """
    UCS: Always expands the node with the lowest cumulative cost.
    Best for weighted graphs where we want minimum study time.
    """
    t0 = time.perf_counter()
    # (cost, node, path_so_far)
    pq = [(0, start, [start])]
    visited = set()
    best_path = []
    nodes_explored = 0

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        nodes_explored += 1
        best_path = path

        if goal and node == goal:
            break

        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(pq, (cost + weight, neighbor, path + [neighbor]))

    elapsed = round((time.perf_counter() - t0) * 1000, 3)
    return {
        "algorithm": "UCS",
        "path": best_path,
        "nodes_explored": nodes_explored,
        "time_ms": elapsed,
        "description": "Expands lowest-cost node first; optimal for weighted graphs. Prioritizes weak/important topics."
    }


# ─────────────────────────────────────────────
# A* Search
# ─────────────────────────────────────────────

def astar_study_plan(graph: dict, start: str, goal: str, weak_topics: list = None):
    """
    A* Search: Uses f(n) = g(n) + h(n) to find optimal path.
    g(n) = actual cost from start, h(n) = heuristic (topic priority).
    """
    if weak_topics is None:
        weak_topics = []
    
    t0 = time.perf_counter()
    # (f, g, node, path)
    open_set = [(0, 0, start, [start])]
    visited = set()
    best_path = []
    nodes_explored = 0

    while open_set:
        f, g, node, path = heapq.heappop(open_set)
        if node in visited:
            continue
        visited.add(node)
        nodes_explored += 1
        best_path = path

        if node == goal:
            break

        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                new_g = g + weight
                h = heuristic(neighbor, goal, weak_topics)
                new_f = new_g + h
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))

    elapsed = round((time.perf_counter() - t0) * 1000, 3)
    return {
        "algorithm": "A*",
        "path": best_path,
        "nodes_explored": nodes_explored,
        "time_ms": elapsed,
        "description": "Combines actual cost + heuristic. Finds optimal path efficiently, prioritizing weak topics."
    }


# ─────────────────────────────────────────────
# HILL CLIMBING
# ─────────────────────────────────────────────

def hill_climbing_study_plan(graph: dict, start: str, weak_topics: list = None, scoring_fn=None):
    """
    Hill Climbing: Greedy local search always moves to best neighbor.
    Uses topic scores to determine 'best' next topic to study.
    """
    if weak_topics is None:
        weak_topics = []

    def default_score(topic):
        # Weak topics score higher (need more attention)
        return 10 if topic in weak_topics else 5

    score_fn = scoring_fn or default_score

    t0 = time.perf_counter()
    current = start
    path = [current]
    visited = {current}
    nodes_explored = 1

    while True:
        neighbors = graph.get(current, [])
        unvisited = [(n, w) for n, w in neighbors if n not in visited]
        if not unvisited:
            break
        # Choose neighbor with highest score (greedy)
        next_node = max(unvisited, key=lambda x: score_fn(x[0]))[0]
        path.append(next_node)
        visited.add(next_node)
        nodes_explored += 1
        current = next_node

    elapsed = round((time.perf_counter() - t0) * 1000, 3)
    return {
        "algorithm": "Hill Climbing",
        "path": path,
        "nodes_explored": nodes_explored,
        "time_ms": elapsed,
        "description": "Greedy local search; always picks the highest-priority topic next. Fast but may miss global optimum."
    }


# ─────────────────────────────────────────────
# STUDY PLAN GENERATOR
# ─────────────────────────────────────────────

def generate_study_plan(subject: str, topics: list, weak_topics: list,
                        exam_days: int = 7, algorithm: str = "A*"):
    """
    Generate a day-wise study plan using the specified search algorithm.
    Returns structured plan with daily allocations.
    """
    if not topics:
        return {"error": "No topics provided"}

    # Build graph
    graph = build_study_graph(topics, weak_topics)
    start = topics[0]
    goal = topics[-1]

    # Run selected algorithm
    algo_map = {
        "BFS": lambda: bfs_study_plan(graph, start),
        "DFS": lambda: dfs_study_plan(graph, start),
        "UCS": lambda: ucs_study_plan(graph, start),
        "A*": lambda: astar_study_plan(graph, start, goal, weak_topics),
        "Hill Climbing": lambda: hill_climbing_study_plan(graph, start, weak_topics),
    }

    result = algo_map.get(algorithm, algo_map["A*"])()
    ordered_topics = result["path"]

    # Distribute topics across exam days
    topics_per_day = max(1, len(ordered_topics) // exam_days)
    daily_plan = []
    
    for day in range(1, exam_days + 1):
        start_idx = (day - 1) * topics_per_day
        end_idx = start_idx + topics_per_day if day < exam_days else len(ordered_topics)
        day_topics = ordered_topics[start_idx:end_idx]
        
        if not day_topics:
            continue
        
        day_hours = sum(2.5 if t in weak_topics else 1.5 for t in day_topics)
        daily_plan.append({
            "day": day,
            "topics": day_topics,
            "hours": round(day_hours, 1),
            "focus": "🔴 Intensive (Weak Topics)" if any(t in weak_topics for t in day_topics) else "🟢 Standard"
        })

    return {
        "subject": subject,
        "algorithm": algorithm,
        "algorithm_info": result,
        "total_topics": len(ordered_topics),
        "exam_days": exam_days,
        "daily_plan": daily_plan,
        "weak_topics_highlighted": weak_topics
    }


def compare_all_algorithms(topics: list, weak_topics: list):
    """Run all algorithms and return comparison metrics."""
    graph = build_study_graph(topics, weak_topics)
    start = topics[0]
    goal = topics[-1] if len(topics) > 1 else topics[0]

    results = []
    results.append(bfs_study_plan(graph, start))
    results.append(dfs_study_plan(graph, start))
    results.append(ucs_study_plan(graph, start))
    results.append(astar_study_plan(graph, start, goal, weak_topics))
    results.append(hill_climbing_study_plan(graph, start, weak_topics))
    
    return results
