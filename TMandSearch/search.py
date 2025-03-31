#!/usr/bin/env python3
"""
search.py

This file demonstrates search algorithms on a grid. It includes:
- A GridWithWeights class for constructing grids with walls and weighted cells.
- Implementations of Breadth-First Search, Dijkstra's (Uniform Cost) Search, and A* Search.
- Functions to draw the grid and reconstruct the found path.

It also provides an example with a 30x30 grid that has barriers and expensive cells.
"""

import collections
import heapq
import math

# -----------------------
# Grid and Utility Functions

class GridWithWeights:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []       # List of impassable cells: [(x,y), ...]
        self.weights = {}     # Dictionary: {(x,y): weight}

    def in_bounds(self, pos):
        (x, y) = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, pos):
        return pos not in self.walls

    def cost(self, from_node, to_node):
        # Default cost is 1 if not in weights.
        return self.weights.get(to_node, 1)

    def neighbors(self, pos):
        (x, y) = pos
        # Four possible moves: right, up, left, down
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        # Filter out positions that are not in bounds or not passable.
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return list(results)

# -----------------------
# Heuristic Functions

def euclidean_heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def manhattan_heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

# -----------------------
# Search Algorithms

def breadth_first_search(graph, start, goal):
    frontier = collections.deque()
    frontier.append(start)
    came_from = {start: None}
    
    while frontier:
        current = frontier.popleft()
        if current == goal:
            break
        for neighbor in graph.neighbors(current):
            if neighbor not in came_from:
                frontier.append(neighbor)
                came_from[neighbor] = current
    return came_from

def dijkstra_search(graph, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = current
    return came_from, cost_so_far

def a_star_search(graph, start, goal):
    # A* search using Euclidean distance as the heuristic
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        current_priority, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + euclidean_heuristic(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
    return came_from, cost_so_far

def a_star_search_custom(graph, start, goal, heuristic_func):
    # A* search that uses a custom heuristic function.
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        current_priority, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, neighbor)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic_func(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current
    return came_from, cost_so_far

# -----------------------
# Grid Drawing and Path Reconstruction

def draw_grid(graph, **kwargs):
    """
    Draws the grid in the terminal.
    Optional keyword arguments:
      - start: starting cell (tuple)
      - goal: goal cell (tuple)
      - path: list of cells that make up the path
      - point_to: dictionary mapping cell -> predecessor (used for visualizing search tree)
    """
    start = kwargs.get('start')
    goal = kwargs.get('goal')
    path = kwargs.get('path', [])
    point_to = kwargs.get('point_to', {})
    
    for y in range(graph.height):
        for x in range(graph.width):
            pos = (x, y)
            if pos == start:
                print(" A ", end="")
            elif pos == goal:
                print(" Z ", end="")
            elif pos in graph.walls:
                print("###", end="")
            elif path and pos in path:
                print(" @ ", end="")
            elif point_to and pos in point_to:
                print(" . ", end="")
            else:
                print("   ", end="")
        print()  # new line at the end of each row

def reconstruct_path(came_from, start, goal):
    """
    Reconstructs the path from start to goal using the came_from dictionary.
    """
    current = goal
    path = []
    while current != start:
        if current not in came_from:
            return []  # no path found
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# -----------------------
# Main block to demonstrate the search algorithms on a 30x30 grid

if __name__ == '__main__':
    # Construct a 30x30 grid.
    g = GridWithWeights(30, 30)
    
    # Add some walls (barriers) in strategic positions.
    g.walls = [(5, i) for i in range(5, 25)] + [(i, 15) for i in range(10, 20)]
    
    # Add weighted (expensive) areas.
    # For example, all cells along the column x=i where y == 10 are given a higher cost.
    expensive_cells = [(i, 10) for i in range(0, 30)]
    for loc in expensive_cells:
        g.weights[loc] = 10

    # Define start and goal positions.
    start = (0, 0)
    goal = (29, 29)

    # -----------------------
    # Breadth-First Search (BFS)
    print("Breadth-First Search (BFS):")
    parents_bfs = breadth_first_search(g, start, goal)
    path_bfs = reconstruct_path(parents_bfs, start, goal)
    draw_grid(g, path=path_bfs, start=start, goal=goal)
    print()

    # -----------------------
    # Dijkstra's (Uniform Cost) Search
    print("Dijkstra's Search:")
    came_from_dij, cost_dij = dijkstra_search(g, start, goal)
    path_dij = reconstruct_path(came_from_dij, start, goal)
    draw_grid(g, path=path_dij, start=start, goal=goal)
    print()

    # -----------------------
    # A* Search with default (Euclidean) heuristic
    print("A* Search with Euclidean Heuristic:")
    came_from_astar, cost_astar = a_star_search(g, start, goal)
    path_astar = reconstruct_path(came_from_astar, start, goal)
    draw_grid(g, path=path_astar, start=start, goal=goal)
    print()

    # -----------------------
    # A* Search with a custom (Manhattan) heuristic
    print("A* Search with Manhattan Heuristic:")
    came_from_astar_manh, cost_astar_manh = a_star_search_custom(g, start, goal, manhattan_heuristic)
    path_astar_manh = reconstruct_path(came_from_astar_manh, start, goal)
    draw_grid(g, path=path_astar_manh, start=start, goal=goal)
    print()
