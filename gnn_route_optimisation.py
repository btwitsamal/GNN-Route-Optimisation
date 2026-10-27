# Name: Amal Thonniparambil Madhusudhanan
# AI individual code (COMP1827 Introduction to Artificial Intelligence)
# Overview: This code implements a GA to solve MOO (multi-objective optimisation), then enhances it using a GNN heuristic
# I BASELINE (GROUP) MODEL:
#  Classes: Route and RouteFinderGA.
#  Time Metric: Relies on static assuming no traffic condition.

# II INDIVIDUAL CONTRIBUTION (GNN ENHANCEMENT):
#  Classes: EnhancedRoute and EnhancedRouteFinderGA.
#  Time Metric: Replaces the static 'time' lookup with a Graph Neural
#  Network (GNN) heuristic, 'get_gnn_time_estimate'.
#The main structure of the Genetic Algorithm and GNN was referenced from the given below public repository:
# Repository Name: [JTexpo]
# GitHub URL: https://jtexpo.github.io/
# I also took help of Gemini to get some of the classes and structure for the code implementation




import random
from typing import List, Dict, Tuple, Optional

city_graph = {
    "Home": [
        {'to': 'Bus_Station', 'mode': 'Walk', 'cost': 0.0, 'time': 29, 'comfort': 2},
        {'to': 'Bus_Station', 'mode': 'Bus', 'cost': 1.73, 'time': 11, 'comfort': 3},
        {'to': 'Bus_Station', 'mode': 'Taxi', 'cost': 9.69, 'time': 5, 'comfort': 8},
        {'to': 'Stadium', 'mode': 'Walk', 'cost': 0.0, 'time': 108, 'comfort': 1},
        {'to': 'Stadium', 'mode': 'Bus', 'cost': 2.4, 'time': 27, 'comfort': 5},
        {'to': 'Stadium', 'mode': 'Taxi', 'cost': 22.91, 'time': 13, 'comfort': 9},
    ],
    "Bus_Station": [
        {'to': 'Train_Station', 'mode': 'Walk', 'cost': 0.0, 'time': 62, 'comfort': 1},
        {'to': 'Train_Station', 'mode': 'Bus', 'cost': 1.98, 'time': 19, 'comfort': 3},
        {'to': 'Train_Station', 'mode': 'Taxi', 'cost': 14.66, 'time': 6, 'comfort': 8},
        {'to': 'Airport', 'mode': 'Walk', 'cost': 0.0, 'time': 40, 'comfort': 1},
        {'to': 'Airport', 'mode': 'Bus', 'cost': 1.82, 'time': 14, 'comfort': 5},
        {'to': 'Airport', 'mode': 'Taxi', 'cost': 11.46, 'time': 7, 'comfort': 9},
    ],
    "Train_Station": [
        {'to': 'City_Center', 'mode': 'Walk', 'cost': 0.0, 'time': 129, 'comfort': 2},
        {'to': 'City_Center', 'mode': 'Bus', 'cost': 2.55, 'time': 35, 'comfort': 4},
        {'to': 'City_Center', 'mode': 'Taxi', 'cost': 25.93, 'time': 13, 'comfort': 9},
        {'to': 'Cafe_district', 'mode': 'Walk', 'cost': 0.0, 'time': 182, 'comfort': 1},
        {'to': 'Cafe_district', 'mode': 'Bus', 'cost': 2.99, 'time': 47, 'comfort': 3},
        {'to': 'Cafe_district', 'mode': 'Taxi', 'cost': 34.83, 'time': 21, 'comfort': 10},
    ],
    "City_Center": [
        {'to': 'University', 'mode': 'Walk', 'cost': 0.0, 'time': 59, 'comfort': 2},
        {'to': 'University', 'mode': 'Bus', 'cost': 1.98, 'time': 16, 'comfort': 4},
        {'to': 'University', 'mode': 'Taxi', 'cost': 14.67, 'time': 9, 'comfort': 10},
        {'to': 'Zoo', 'mode': 'Walk', 'cost': 0.0, 'time': 81, 'comfort': 2},
        {'to': 'Zoo', 'mode': 'Bus', 'cost': 2.16, 'time': 20, 'comfort': 5},
        {'to': 'Zoo', 'mode': 'Taxi', 'cost': 18.21, 'time': 8, 'comfort': 10},
    ],
    "University": [
        {'to': 'Library', 'mode': 'Walk', 'cost': 0.0, 'time': 27, 'comfort': 1},
        {'to': 'Library', 'mode': 'Bus', 'cost': 1.71, 'time': 10, 'comfort': 4},
        {'to': 'Library', 'mode': 'Taxi', 'cost': 9.24, 'time': 6, 'comfort': 8},
        {'to': 'Hospital', 'mode': 'Walk', 'cost': 0.0, 'time': 101, 'comfort': 3},
        {'to': 'Hospital', 'mode': 'Bus', 'cost': 2.32, 'time': 28, 'comfort': 3},
        {'to': 'Hospital', 'mode': 'Taxi', 'cost': 21.49, 'time': 12, 'comfort': 9},
    ],
    "Library": [
        {'to': 'Shopping_Mall', 'mode': 'Walk', 'cost': 0.0, 'time': 93, 'comfort': 3},
        {'to': 'Shopping_Mall', 'mode': 'Bus', 'cost': 2.25, 'time': 23, 'comfort': 5},
        {'to': 'Shopping_Mall', 'mode': 'Taxi', 'cost': 19.93, 'time': 11, 'comfort': 10},
    ],
    "Shopping_Mall": [
        {'to': 'Hospital', 'mode': 'Walk', 'cost': 0.0, 'time': 163, 'comfort': 2},
        {'to': 'Hospital', 'mode': 'Bus', 'cost': 2.85, 'time': 43, 'comfort': 3},
        {'to': 'Hospital', 'mode': 'Taxi', 'cost': 31.99, 'time': 20, 'comfort': 8},
        {'to': 'Museum', 'mode': 'Walk', 'cost': 0.0, 'time': 27, 'comfort': 3},
        {'to': 'Museum', 'mode': 'Bus', 'cost': 1.71, 'time': 11, 'comfort': 4},
        {'to': 'Museum', 'mode': 'Taxi', 'cost': 9.27, 'time': 7, 'comfort': 9},
    ],
    "Hospital": [
        {'to': 'Park_Entrance', 'mode': 'Walk', 'cost': 0.0, 'time': 58, 'comfort': 1},
        {'to': 'Park_Entrance', 'mode': 'Bus', 'cost': 1.95, 'time': 18, 'comfort': 5},
        {'to': 'Park_Entrance', 'mode': 'Taxi', 'cost': 13.99, 'time': 9, 'comfort': 9},
        {'to': 'Shopping_Mall', 'mode': 'Walk', 'cost': 0.0, 'time': 30, 'comfort': 2},
        {'to': 'Shopping_Mall', 'mode': 'Bus', 'cost': 1.72, 'time': 8, 'comfort': 5},
        {'to': 'Shopping_Mall', 'mode': 'Taxi', 'cost': 9.33, 'time': 5, 'comfort': 9},
    ],
    "Park_Entrance": [
        {'to': 'Museum', 'mode': 'Walk', 'cost': 0.0, 'time': 30, 'comfort': 3},
        {'to': 'Museum', 'mode': 'Bus', 'cost': 1.75, 'time': 8, 'comfort': 5},
        {'to': 'Museum', 'mode': 'Taxi', 'cost': 9.9, 'time': 6, 'comfort': 8},
        {'to': 'River_Dock', 'mode': 'Walk', 'cost': 0.0, 'time': 50, 'comfort': 2},
        {'to': 'River_Dock', 'mode': 'Bus', 'cost': 1.9, 'time': 16, 'comfort': 5},
        {'to': 'River_Dock', 'mode': 'Taxi', 'cost': 13.07, 'time': 8, 'comfort': 8},
    ],
    "Museum": [
        {'to': 'Cinema', 'mode': 'Walk', 'cost': 0.0, 'time': 83, 'comfort': 2},
        {'to': 'Cinema', 'mode': 'Bus', 'cost': 2.15, 'time': 20, 'comfort': 3},
        {'to': 'Cinema', 'mode': 'Taxi', 'cost': 18.07, 'time': 11, 'comfort': 9},
        {'to': 'Suburbs', 'mode': 'Walk', 'cost': 0.0, 'time': 82, 'comfort': 1},
        {'to': 'Suburbs', 'mode': 'Bus', 'cost': 2.16, 'time': 22, 'comfort': 5},
        {'to': 'Suburbs', 'mode': 'Taxi', 'cost': 18.28, 'time': 12, 'comfort': 9},
    ],
    "Cinema": [
        {'to': 'Gym', 'mode': 'Walk', 'cost': 0.0, 'time': 162, 'comfort': 2},
        {'to': 'Gym', 'mode': 'Bus', 'cost': 2.82, 'time': 44, 'comfort': 3},
        {'to': 'Gym', 'mode': 'Taxi', 'cost': 31.44, 'time': 17, 'comfort': 10},
        {'to': 'University', 'mode': 'Walk', 'cost': 0.0, 'time': 170, 'comfort': 1},
        {'to': 'University', 'mode': 'Bus', 'cost': 2.9, 'time': 45, 'comfort': 3},
        {'to': 'University', 'mode': 'Taxi', 'cost': 32.98, 'time': 21, 'comfort': 9},
    ],
    "Gym": [
        {'to': 'Cafe_district', 'mode': 'Walk', 'cost': 0.0, 'time': 169, 'comfort': 3},
        {'to': 'Cafe_district', 'mode': 'Bus', 'cost': 2.89, 'time': 43, 'comfort': 3},
        {'to': 'Cafe_district', 'mode': 'Taxi', 'cost': 32.8, 'time': 18, 'comfort': 8},
        {'to': 'Stadium', 'mode': 'Walk', 'cost': 0.0, 'time': 78, 'comfort': 3},
        {'to': 'Stadium', 'mode': 'Bus', 'cost': 2.14, 'time': 23, 'comfort': 3},
        {'to': 'Stadium', 'mode': 'Taxi', 'cost': 17.9, 'time': 8, 'comfort': 10},
    ],
    "Cafe_district": [
        {'to': 'River_Dock', 'mode': 'Walk', 'cost': 0.0, 'time': 79, 'comfort': 3},
        {'to': 'River_Dock', 'mode': 'Bus', 'cost': 2.12, 'time': 22, 'comfort': 4},
        {'to': 'River_Dock', 'mode': 'Taxi', 'cost': 17.43, 'time': 9, 'comfort': 9},
        {'to': 'University', 'mode': 'Walk', 'cost': 0.0, 'time': 145, 'comfort': 2},
        {'to': 'University', 'mode': 'Bus', 'cost': 2.7, 'time': 39, 'comfort': 3},
        {'to': 'University', 'mode': 'Taxi', 'cost': 28.96, 'time': 17, 'comfort': 10},
    ],
    "River_Dock": [
        {'to': 'Tech_Park', 'mode': 'Walk', 'cost': 0.0, 'time': 125, 'comfort': 1},
        {'to': 'Tech_Park', 'mode': 'Bus', 'cost': 2.52, 'time': 35, 'comfort': 4},
        {'to': 'Tech_Park', 'mode': 'Taxi', 'cost': 25.42, 'time': 15, 'comfort': 8},
        {'to': 'Stadium', 'mode': 'Walk', 'cost': 0.0, 'time': 92, 'comfort': 1},
        {'to': 'Stadium', 'mode': 'Bus', 'cost': 2.24, 'time': 23, 'comfort': 5},
        {'to': 'Stadium', 'mode': 'Taxi', 'cost': 19.71, 'time': 11, 'comfort': 10},
    ],
    "Tech_Park": [
        {'to': 'Hotel', 'mode': 'Walk', 'cost': 0.0, 'time': 44, 'comfort': 1},
        {'to': 'Hotel', 'mode': 'Bus', 'cost': 1.85, 'time': 13, 'comfort': 5},
        {'to': 'Hotel', 'mode': 'Taxi', 'cost': 12.0, 'time': 8, 'comfort': 8},
        {'to': 'Cafe_district', 'mode': 'Walk', 'cost': 0.0, 'time': 153, 'comfort': 3},
        {'to': 'Cafe_district', 'mode': 'Bus', 'cost': 2.77, 'time': 40, 'comfort': 4},
        {'to': 'Cafe_district', 'mode': 'Taxi', 'cost': 30.43, 'time': 18, 'comfort': 9},
    ],
    "Hotel": [
        {'to': 'Airport', 'mode': 'Walk', 'cost': 0.0, 'time': 144, 'comfort': 3},
        {'to': 'Airport', 'mode': 'Bus', 'cost': 2.69, 'time': 40, 'comfort': 4},
        {'to': 'Airport', 'mode': 'Taxi', 'cost': 28.89, 'time': 17, 'comfort': 8},
        {'to': 'Cafe_district', 'mode': 'Walk', 'cost': 0.0, 'time': 77, 'comfort': 1},
        {'to': 'Cafe_district', 'mode': 'Bus', 'cost': 2.12, 'time': 22, 'comfort': 5},
        {'to': 'Cafe_district', 'mode': 'Taxi', 'cost': 17.42, 'time': 9, 'comfort': 8},
    ],
    "Airport": [
        {'to': 'Stadium', 'mode': 'Walk', 'cost': 0.0, 'time': 58, 'comfort': 2},
        {'to': 'Stadium', 'mode': 'Bus', 'cost': 1.95, 'time': 15, 'comfort': 3},
        {'to': 'Stadium', 'mode': 'Taxi', 'cost': 13.9, 'time': 6, 'comfort': 9},
        {'to': 'Bus_Station', 'mode': 'Walk', 'cost': 0.0, 'time': 47, 'comfort': 1},
        {'to': 'Bus_Station', 'mode': 'Bus', 'cost': 1.88, 'time': 12, 'comfort': 3},
        {'to': 'Bus_Station', 'mode': 'Taxi', 'cost': 12.66, 'time': 5, 'comfort': 9},
    ],
    "Stadium": [
        {'to': 'Zoo', 'mode': 'Walk', 'cost': 0.0, 'time': 38, 'comfort': 2},
        {'to': 'Zoo', 'mode': 'Bus', 'cost': 1.8, 'time': 10, 'comfort': 4},
        {'to': 'Zoo', 'mode': 'Taxi', 'cost': 11.04, 'time': 8, 'comfort': 10},
        {'to': 'Home', 'mode': 'Walk', 'cost': 0.0, 'time': 68, 'comfort': 3},
        {'to': 'Home', 'mode': 'Bus', 'cost': 2.05, 'time': 17, 'comfort': 3},
        {'to': 'Home', 'mode': 'Taxi', 'cost': 16.08, 'time': 11, 'comfort': 10},
    ],
    "Zoo": [
        {'to': 'Suburbs', 'mode': 'Walk', 'cost': 0.0, 'time': 60, 'comfort': 3},
        {'to': 'Suburbs', 'mode': 'Bus', 'cost': 2.0, 'time': 16, 'comfort': 5},
        {'to': 'Suburbs', 'mode': 'Taxi', 'cost': 14.91, 'time': 10, 'comfort': 10},
    ],
    "Suburbs": [
        {'to': 'Home', 'mode': 'Walk', 'cost': 0.0, 'time': 132, 'comfort': 2},
        {'to': 'Home', 'mode': 'Bus', 'cost': 2.58, 'time': 36, 'comfort': 5},
        {'to': 'Home', 'mode': 'Taxi', 'cost': 26.67, 'time': 15, 'comfort': 8},
        {'to': 'Stadium', 'mode': 'Walk', 'cost': 0.0, 'time': 82, 'comfort': 1},
        {'to': 'Stadium', 'mode': 'Bus', 'cost': 2.17, 'time': 22, 'comfort': 4},
        {'to': 'Stadium', 'mode': 'Taxi', 'cost': 18.44, 'time': 10, 'comfort': 9},
    ],
}


class Route:
    def __init__(self, path: List[Dict]):
        """
        A Route represents a complete journey from start to destination.
        path: List of transport segments, each segment is a dict from city_graph
        """
        self.path = path
        self.total_cost = 0
        self.total_time = 0
        self.total_comfort = 0
        self.fitness = 0
        self.calculate_totals()

    def calculate_totals(self):
        """Calculate total cost, time, and average comfort for the route"""
        self.total_cost = sum(segment["cost"] for segment in self.path)
        self.total_time = sum(segment["time"] for segment in self.path)
        if self.path:
            self.total_comfort = sum(segment["comfort"] for segment in self.path) / len(self.path)
        else:
            self.total_comfort = 0

    def calculate_fitness(self, weights: Dict[str, float] = None):
        """
        Calculate fitness using weighted sum approach
        Lower fitness is better (we're minimizing)
        """
        if weights is None:
            # FIX: Hybrid Weights to force a MIX of transport modes
            weights = {"time": 0.46, "cost": 0.44, "comfort": 0.1}

        # Normalize and weight the objectives
        # Max time ~300 mins, Max cost ~£100 (assuming multiple taxi rides)
        normalized_time = self.total_time / 300
        normalized_cost = self.total_cost / 100
        normalized_discomfort = (10 - self.total_comfort) / 10

        self.fitness = (
                weights["time"] * normalized_time +
                weights["cost"] * normalized_cost +
                weights["comfort"] * normalized_discomfort
        )
        return self.fitness

    def __str__(self):
        if not self.path:
            return "Empty Route"

        # Determine the starting node for display
        start_node = self.path[0].get('from', 'Home')

        route_str = f"Route: {start_node}"

        for segment in self.path:
            route_str += f" -> {segment['to']} ({segment['mode']})"

        route_str += f"\nCost: £{self.total_cost:.2f}, Time: {self.total_time:.2f}min, Comfort: {self.total_comfort:.1f}/10, Fitness: {self.fitness:.3f}"
        return route_str


class RouteFinderGA:
    def __init__(self, graph: Dict, population_size: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.graph = graph
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[Route] = []

    def find_all_paths(self, start: str, end: str, max_depth: int = 10) -> List[List[Dict]]:
        """Find all possible paths from start to end using DFS"""
        all_paths = []

        def dfs(current: str, path: List[Dict], visited: set, depth: int):
            if depth > max_depth:
                return
            if current == end:
                # Inject 'from' field into each segment for consistency
                complete_path = []
                previous_node = start
                for segment in path:
                    segment_copy = segment.copy()
                    segment_copy['from'] = previous_node
                    complete_path.append(segment_copy)
                    previous_node = segment_copy['to']
                all_paths.append(complete_path)
                return

            if current not in self.graph:
                return

            visited.add(current)

            for connection in self.graph[current]:
                next_node = connection["to"]
                # For simple route finding, strict non-revisit is used for path generation
                if next_node not in visited:
                    new_path = path + [connection]
                    dfs(next_node, new_path, visited.copy(), depth + 1)

        dfs(start, [], set(), 0)
        return all_paths

    def initialize_population(self, start: str, end: str):
        """Initialize population with valid routes from start to end"""
        all_paths = self.find_all_paths(start, end)

        if not all_paths:
            print("Warning: No paths found for initialization.")
            return

        # If we have enough paths, use them directly
        if len(all_paths) >= self.population_size:
            selected_paths = random.sample(all_paths, self.population_size)
        else:
            # Duplicate and mutate some paths to fill population
            selected_paths = all_paths.copy()
            while len(selected_paths) < self.population_size:
                # Use deepcopy to prevent shared mutation effects
                selected_paths.append([seg.copy() for seg in random.choice(all_paths)])

        # IMPORTANT: This line uses the base Route class
        self.population = [Route(path) for path in selected_paths]

    def select_parents(self) -> Tuple[Route, Route]:
        """Select two parents using tournament selection"""
        if len(self.population) < 3:
            return random.choice(self.population), random.choice(self.population)

        tournament_size = 3
        tournament1 = random.sample(self.population, tournament_size)
        tournament2 = random.sample(self.population, tournament_size)

        parent1 = min(tournament1, key=lambda x: x.fitness)
        parent2 = min(tournament2, key=lambda x: x.fitness)

        return parent1, parent2

    def crossover(self, parent1: Route, parent2: Route) -> Tuple[Route, Route]:
        """Perform crossover between two parents to create two children"""
        # Ensure we have the appropriate class type for construction later
        RouteClass = type(parent1)

        if random.random() > self.crossover_rate or len(parent1.path) < 2 or len(parent2.path) < 2:
            return parent1, parent2

        # Extract nodes to find common crossover point
        parent1_path_nodes = [parent1.path[0]['from']] + [segment['to'] for segment in parent1.path]
        parent2_path_nodes = [parent2.path[0]['from']] + [segment['to'] for segment in parent2.path]

        common_nodes = set(parent1_path_nodes) & set(parent2_path_nodes)
        if not common_nodes:
            return parent1, parent2

        # Remove start and end nodes as crossover points to simplify path merging
        start_node = parent1_path_nodes[0]
        end_node = parent1_path_nodes[-1]
        crossover_candidates = list(common_nodes - {start_node, end_node})

        if not crossover_candidates:
            return parent1, parent2

        crossover_point = random.choice(crossover_candidates)

        # Helper function to find the index of the segment that starts/ends at the crossover point
        def get_index_at_node(path, node):
            # Find the index *of the segment leading to* the crossover point
            for i, segment in enumerate(path):
                if segment['to'] == node:
                    return i
            return -1

        # Find the split indices (index of the segment *leading into* crossover_point)
        split1 = get_index_at_node(parent1.path, crossover_point)
        split2 = get_index_at_node(parent2.path, crossover_point)

        if split1 == -1 or split2 == -1:
            return parent1, parent2  # Safety check

        # Create children paths:
        # Child 1: P1 (Start to Crossover) + P2 (Crossover to End)
        child1_path = parent1.path[:split1 + 1] + parent2.path[split2 + 1:]

        # Child 2: P2 (Start to Crossover) + P1 (Crossover to End)
        child2_path = parent2.path[:split2 + 1] + parent1.path[split1 + 1:]

        # Repair 'from' nodes due to merging:
        # The segment *after* the split point needs its 'from' node reset to the crossover point
        if split1 + 1 < len(child1_path):
            child1_path[split1 + 1]['from'] = crossover_point
        if split2 + 1 < len(child2_path):
            child2_path[split2 + 1]['from'] = crossover_point

        # Use deepcopy for all segments and validate path to remove duplicates
        child1_path = [seg.copy() for seg in self._validate_path(child1_path)]
        child2_path = [seg.copy() for seg in self._validate_path(child2_path)]

        # Use the correct RouteClass (Route or EnhancedRoute) to instantiate children
        return RouteClass(child1_path), RouteClass(child2_path)

    def mutate(self, route: Route) -> Route:
        """Mutate a route by changing one transport mode"""
        RouteClass = type(route)

        if random.random() > self.mutation_rate or not route.path:
            return route

        # Choose a random segment to mutate
        mutate_index = random.randint(0, len(route.path) - 1)
        current_segment = route.path[mutate_index]

        # Safely determine the 'from' node for the current segment
        from_node = current_segment.get('from', route.path[mutate_index - 1]['to'] if mutate_index > 0 else 'Home')

        # Get alternative transport options for this segment
        if from_node in self.graph:
            alternatives = [conn for conn in self.graph[from_node]
                            if conn['to'] == current_segment['to']]

            if len(alternatives) > 1:
                # Choose a different transport mode
                new_segment = random.choice([alt for alt in alternatives
                                             if alt['mode'] != current_segment['mode']])

                # Create new path with mutated segment
                new_path = route.path.copy()

                # IMPORTANT: Create a copy of the new segment dictionary and inject 'from' field
                mutated_segment = new_segment.copy()
                mutated_segment['from'] = from_node

                new_path[mutate_index] = mutated_segment

                # Return new instance of the correct class
                return RouteClass(new_path)

        return route

    def _validate_path(self, path: List[Dict]) -> List[Dict]:
        """Ensure path is valid (no gaps, reaches destination) - mainly removes duplicates"""
        if not path:
            return path

        # Remove duplicate consecutive segments (retaining path integrity for complex ops)
        valid_path = []
        for segment in path:
            # Check if the segment's 'to' node is the same as the previous valid segment's 'to'
            if not valid_path or segment['to'] != valid_path[-1]['to']:
                valid_path.append(segment)
            # We defer path connectivity validation to path generation for simplicity
        return valid_path

    def evolve(self, start: str, end: str, generations: int = 100,
               weights: Dict[str, float] = None) -> Route:
        """Run the genetic algorithm to find optimal route"""

        # Initialize population
        self.initialize_population(start, end)

        if not self.population:
            print("Evolution aborted: Initial population is empty.")
            return None

        # Calculate initial fitness
        for route in self.population:
            route.calculate_fitness(weights)

        best_route = min(self.population, key=lambda x: x.fitness)

        for generation in range(generations):
            new_population = []

            # Elitism: keep the best individual
            new_population.append(best_route)

            # Create new population
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()

                # Crossover handles the Path finding and returns the appropriate RouteClass objects
                child1, child2 = self.crossover(parent1, parent2)

                # Mutate children
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # Recalculate fitness (totals are recalculated during object initialization)
                child1.calculate_fitness(weights)
                child2.calculate_fitness(weights)

                new_population.extend([child1, child2])

            # Trim if we have too many
            self.population = new_population[:self.population_size]

            # Update best route
            current_best = min(self.population, key=lambda x: x.fitness)
            if current_best.fitness < best_route.fitness:
                best_route = current_best

            # Print progress
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {best_route.fitness:.3f}")

        return best_route


# -------------------------------------------------------------
# INDIVIDUAL CONTRIBUTION: GNN ENHANCEMENT (SIMULATED)
# -------------------------------------------------------------

simulated_node_features = {
    'Home': [1.0, 5.0, 0.6], 'Bus_Station': [1.5, 4.5, 0.8],
    'Train_Station': [2.0, 3.5, 0.9], 'City_Center': [3.0, 3.0, 0.95],
    'University': [4.0, 2.0, 0.7], 'Library': [4.5, 1.5, 0.6],
    'Shopping_Mall': [5.0, 1.0, 0.8], 'Hospital': [3.5, 1.0, 0.7],
    'Park_Entrance': [2.5, 0.5, 0.5], 'Museum': [3.0, 0.0, 0.6],
    'Cinema': [5.5, 0.0, 0.8], 'Gym': [6.0, 0.5, 0.9],
    'Cafe_district': [5.5, 1.5, 0.85], 'River_Dock': [6.5, 2.0, 0.6],
    'Tech_Park': [7.0, 3.0, 0.7], 'Hotel': [6.0, 4.0, 0.8],
    'Airport': [7.5, 5.0, 0.5], 'Stadium': [6.0, 5.5, 0.7],
    'Zoo': [5.0, 5.0, 0.6], 'Suburbs': [4.5, 6.0, 0.4]
}


def get_gnn_time_estimate(from_node: str, to_node: str, base_time: float, mode: str) -> float:
    """
    SIMULATED GNN function: Replaces the static 'time' with a dynamic prediction
    based on the GNN's 'learned' topological and contextual features.

    This function demonstrates the architectural replacement.
    """

    # 1. GNN's focus: Traffic Index at the destination node
    traffic_index = simulated_node_features.get(to_node, [0, 0, 0.5])[2]

    # 2. Simulated Congestion Sensitivity (Simulates GNN learning mode-specific effects)
    congestion_sensitivity = {
        'Walk': 1.0,
        'Bus': 1.25,  # Highly affected
        'Taxi': 1.15,  # Affected
    }

    # 3. Calculation: Base_Time * [1 + (Traffic_Index * Sensitivity)] * Admissibility Factor
    dynamic_multiplier = 1.0 + (traffic_index * congestion_sensitivity.get(mode, 1.0))

    # The 0.98 factor simulates the GNN being trained with Asymmetric Loss
    # to be slightly admissible (never overestimate by much).
    gnn_estimated_time = base_time * dynamic_multiplier * 0.98

    # Ensure time is always positive (Physical Constraint)
    return max(1.0, gnn_estimated_time)


class EnhancedRoute(Route):
    """
    Individual Report: Enhanced Route class that uses the GNN model
    for the time objective.
    """

    def calculate_totals(self):
        """
        Overrides the base method to use the GNN for time estimation.
        Time is now dynamic and intelligent (GNN-based).
        """
        self.total_cost = sum(segment["cost"] for segment in self.path)

        gnn_estimated_times = []
        for i, segment in enumerate(self.path):

            # FIX FOR KEY ERROR: Determine 'from' node by looking at the previous segment's 'to'
            from_node = segment.get('from')
            if from_node is None:
                from_node = self.path[i - 1]['to'] if i > 0 else 'Home'

            # Pass the static time as a baseline input to the simulated GNN
            base_time = segment.get("time", 1.0)
            mode = segment.get("mode", 'Walk')

            # Use the GNN function to get the dynamic, intelligent time estimate
            dynamic_time = get_gnn_time_estimate(from_node, segment['to'], base_time, mode)
            gnn_estimated_times.append(dynamic_time)

        self.total_time = sum(gnn_estimated_times)

        # Comfort calculation remains the same
        if self.path:
            self.total_comfort = sum(segment["comfort"] for segment in self.path) / len(self.path)
        else:
            self.total_comfort = 0


class EnhancedRouteFinderGA(RouteFinderGA):
    """
    Individual Report: Enhanced GA using the GNN-based Route class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Rebind crossover and mutate to use enhanced versions that return EnhancedRoute
        self.crossover = self._enhanced_crossover
        self.mutate = self._enhanced_mutate

    def initialize_population(self, start: str, end: str):
        """
        Overrides initialization to use the EnhancedRoute class (with GNN logic)
        """
        all_paths = self.find_all_paths(start, end)

        if not all_paths:
            print("Warning: No paths found for initialization.")
            return

        if len(all_paths) >= self.population_size:
            selected_paths = random.sample(all_paths, self.population_size)
        else:
            selected_paths = all_paths.copy()
            while len(selected_paths) < self.population_size:
                selected_paths.append([seg.copy() for seg in random.choice(all_paths)])

        # Use the GNN-enabled route class
        self.population = [EnhancedRoute(path) for path in selected_paths]

    def _enhanced_crossover(self, parent1: EnhancedRoute, parent2: EnhancedRoute) -> Tuple[
        EnhancedRoute, EnhancedRoute]:
        """Ensures crossover returns EnhancedRoute objects using the logic from the base class"""

        child1_base, child2_base = super().crossover(parent1, parent2)

        # Return new EnhancedRoute instances based on the paths found
        return EnhancedRoute(child1_base.path), EnhancedRoute(child2_base.path)

    def _enhanced_mutate(self, route: EnhancedRoute) -> EnhancedRoute:
        """Ensures mutate returns EnhancedRoute objects using the logic from the base class"""

        mutated_route_base = super().mutate(route)

        # Return new EnhancedRoute instance based on the path found
        return EnhancedRoute(mutated_route_base.path)


# -------------------------------------------------------------
# EXAMPLE USAGE AND DEMONSTRATION
# -------------------------------------------------------------

if __name__ == "__main__":

    # --- BASELINE GROUP EXECUTION ---
    print("=" * 70)
    print("BASELINE GROUP EXECUTION (STATIC TIME) ")

    ga_base = RouteFinderGA(city_graph, population_size=30, mutation_rate=0.2)
    start_location = "Home"
    end_location = "River_Dock"
    weights_base = {"time": 0.46, "cost": 0.44, "comfort": 0.1}

    print(f"\nFinding optimal route from {start_location} to {end_location} with baseline model...\n")
    best_route_base = ga_base.evolve(start_location, end_location, generations=50, weights=weights_base)

    if best_route_base:
        print("\n BEST ROUTE FOUND BY BASELINE GA")
        print(best_route_base)

    print("\n" + "=" * 70 + "\n")

    # --- INDIVIDUAL GNN-ENHANCED EXECUTION ---
    print("INDIVIDUAL REPORT: GNN-ENHANCED EXECUTION (DYNAMIC TIME) ")

    ga_enhanced = EnhancedRouteFinderGA(city_graph, population_size=30, mutation_rate=0.2)

    # Focus on comfort (high weight for discomfort penalty) to show sensitivity to GNN time estimates
    weights_comfort = {"time": 0.2, "cost": 0.1, "comfort": 0.7}

    print(f"\nRunning GNN-Enhanced GA for route from {start_location} to {end_location} (Prioritizing Comfort)...\n")
    best_route_gnn = ga_enhanced.evolve(start_location, end_location, generations=50, weights=weights_comfort)

    if best_route_gnn:
        print("\nBEST ROUTE FOUND BY GNN-ENHANCED GA (h_GNN) ")
        print(best_route_gnn)