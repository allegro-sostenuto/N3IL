import itertools
import random
import math
import matplotlib.pyplot as plt
from typing import Tuple, List, Set, Callable

Point = Tuple[int, int]


def slope(p1: Point, p2: Point) -> QQ | str:
    """
    Calculate the slope between two points using exact rational arithmetic.
    
    Args:
        p1: First point (x1, y1).
        p2: Second point (x2, y2).
    
    Returns:
        QQ representing slope, or 'inf' if line is vertical.
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return 'inf'
    return QQ(dy) / QQ(dx)


def are_collinear(p1: Point, p2: Point, p3: Point) -> bool:
    """
    Check if three points are collinear using slope comparisons.
    
    Args:
        p1, p2, p3: Points to test.
    
    Returns:
        True if collinear, False otherwise.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (y1 - y2) * (x1 - x3) == (y1 - y3) * (x1 - x2)


def is_no_three_in_line(points: Set[Point]) -> bool:
    """
    Determine if a set of points has no three collinear.
    
    Args:
        points: Set of 2D points.
    
    Returns:
        True if no three points are collinear, False otherwise.
    """
    n = len(points)
    points = list(points)
    for i in range(n):
        slopes = {}
        for j in range(n):
            if i == j:
                continue
            s = slope(points[i], points[j])
            if s in slopes:
                if are_collinear(points[i], points[slopes[s]], points[j]):
                    return False
            slopes[s] = j
    return True


def plot_no_three_in_line(points: List[Point], n: int | None = None, title: str = "No-3-in-line Set") -> None:
    """
    Plot a 2D grid of points with no-three-in-line constraint.
    
    Args:
        points: List of points to plot.
        n: Optional grid size (autodetected if None).
        title: Plot title.
    """
    if not points:
        print("No points to plot.")
        return

    xs, ys = zip(*points)
    if n is None:
        n = max(max(xs), max(ys)) + 1

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=100, c='blue', edgecolors='black')
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-1, n)
    plt.ylim(-1, n)
    plt.show()


def is_complete(points: Set[Point], all_points: List[Point]) -> bool:
    """
    Check if a set is a maximal no-three-in-line configuration.
    
    Args:
        points: Current point set.
        all_points: Full list of points in the grid.
    
    Returns:
        True if the set is complete and valid, False otherwise.
    """
    if len(points) < 3:
        return False
    if not is_no_three_in_line(points):
        print("Configuration has collinear triples.")
        return False
    for p in all_points:
        if p not in points and is_no_three_in_line(points | {p}):
            return False
    return True


def greedy(n: int, priority: Callable[[Point], float]) -> List[Point]:
    """
    Build a no-three-in-line set greedily based on priority.
    
    Args:
        n: Grid size.
        priority: Function assigning scores to points.
    
    Returns:
        Sorted list of selected points.
    """
    all_points = [(x, y) for x in range(n) for y in range(n)]
    all_points.sort(key=priority, reverse=True)
    current_set: Set[Point] = set()
    for p in all_points:
        trial_set = current_set | {p}
        if is_no_three_in_line(trial_set):
            current_set = trial_set
    return sorted(current_set)


def pnorm_circle_priority(n: int, p: float = 4, radius_scale: float = 0.9) -> Callable[[Point], float]:
    """
    Generate a circular shell priority using p-norm distance.
    
    Args:
        n: Grid size.
        p: Norm power (e.g., 2 for Euclidean).
        radius_scale: Target radius in normalized space.
    
    Returns:
        Priority function for use in greedy().
    """
    def priority(point: Point) -> float:
        x, y = point
        nx = (2 * x / n) - 1
        ny = (2 * y / n) - 1
        norm = (abs(nx)**p + abs(ny)**p)**(1/p)
        noise = random.uniform(-0.1, 0.1)
        return -abs(norm - radius_scale) + noise
    return priority


def square_shell_priority(n: int, sharpness: float = 8, radius_scale: float = 0.95) -> Callable[[Point], float]:
    """
    Generate a priority favoring square-edge shells.
    
    Args:
        n: Grid size.
        sharpness: Dropoff strength from edge.
        radius_scale: Desired distance from center to edge.
    
    Returns:
        Priority function for greedy().
    """
    def priority(point: Point) -> float:
        x, y = point
        nx = (2 * x / n) - 1
        ny = (2 * y / n) - 1
        norm = max(abs(nx), abs(ny))
        score = -abs(norm - radius_scale)**sharpness
        noise = random.uniform(-0.1, 0.1)
        return score + noise
    return priority


def square_corner_priority(n: int, sharpness: float = 8, radius_scale: float = 0.95) -> Callable[[Point], float]:
    """
    Generate a priority function favoring square corners.
    
    Args:
        n: Grid size.
        sharpness: Dropoff strength.
        radius_scale: Desired corner proximity.
    
    Returns:
        Priority function for greedy().
    """
    def priority(point: Point) -> float:
        x, y = point
        nx = x / n
        ny = y / n
        norm = max(nx, ny)
        score = -abs(norm - radius_scale)**sharpness
        noise = random.uniform(-0.1, 0.1)
        return score + noise
    return priority

def random_priority(n: int) -> Callable[[Point], float]: 
    """
    Generate a random priority function.
    
    Args:
        n: Grid size.
    
    Returns:
        Random priority function for greedy().
    """
    def priority(point: Point) -> float:
        return random.uniform(0, 1)
    return priority