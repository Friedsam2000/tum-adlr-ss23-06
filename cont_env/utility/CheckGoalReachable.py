# CheckGoalReachable.py
import heapq

def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def a_star_search(start, goal, obstacles, grid_size):
    start, goal = tuple(start), tuple(goal)
    frontier = []
    heapq.heappush(frontier, (0, start))
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            return True

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_position = (current[0] + dx, current[1] + dy)

            if (next_position[0] < 0 or next_position[0] >= grid_size[0]
                    or next_position[1] < 0 or next_position[1] >= grid_size[1]
                    or next_position in obstacles):
                continue

            new_cost = cost_so_far[current] + 1
            if next_position not in cost_so_far or new_cost < cost_so_far[next_position]:
                cost_so_far[next_position] = new_cost
                priority = new_cost + heuristic(goal, next_position)
                heapq.heappush(frontier, (priority, next_position))

    return False
