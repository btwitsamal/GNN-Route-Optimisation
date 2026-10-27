# GNN-Enhanced Multi-Objective Route Optimisation

A Python **genetic algorithm (GA)** that finds the best route across a 20-location city network, balancing **travel time, cost and comfort** across three transport modes: walk, bus and taxi.

The project then extends the GA with a **GNN-style, traffic-aware time heuristic**. It replaces fixed journey times with dynamic estimates based on traffic at each location.

> Built for **COMP1827 Introduction to Artificial Intelligence** at the University of Greenwich.
> The baseline GA was group work. The GNN enhancement was my individual contribution.

---

## Features

- Multi-objective optimisation of time, cost and comfort using a weighted-sum fitness function
- Hybrid routes that mix walking, bus and taxi segments
- Classic GA operators: tournament selection, crossover, mutation and elitism
- A drop-in time estimator (`get_gnn_time_estimate`) that makes journey times traffic-aware
- Runs with plain Python, with no external libraries

---

## How it works

### 1. Baseline GA (group work): `Route`, `RouteFinderGA`

| Step | What it does |
|------|--------------|
| **Chromosome** | A route, stored as a list of segments (`from`, `to`, `mode`, `cost`, `time`, `comfort`) |
| **Initial population** | Depth-first search finds every valid path from start to destination (up to 10 hops) |
| **Selection** | Tournament selection (size 3) |
| **Crossover** | Two parent routes swap their tails at a location they both pass through |
| **Mutation** | Switches one segment to a different transport mode (e.g. bus → taxi) |
| **Elitism** | The best route is always carried into the next generation |

**Fitness** (lower is better):

```
fitness = w_time × (time / 300) + w_cost × (cost / 100) + w_comfort × ((10 − comfort) / 10)
```

### 2. GNN enhancement (individual work): `EnhancedRoute`, `EnhancedRouteFinderGA`

In the baseline, every journey takes a fixed time, whatever the traffic.
The enhanced version replaces that fixed value with a dynamic estimate:

```
estimated_time = base_time × (1 + traffic_index × mode_sensitivity) × 0.98
```

- **traffic_index** (0–1) comes from each location's feature vector
- **mode_sensitivity** reflects how much traffic affects each mode: Walk 1.0, Taxi 1.15, Bus 1.25
- **0.98** keeps estimates slightly optimistic, like a model trained not to overestimate

`EnhancedRoute` uses these estimates when it totals a route's time. `EnhancedRouteFinderGA` makes sure every new route created during evolution uses the enhanced version.

> **Note:** the GNN is **simulated**. The location features and mode sensitivities are hand-set values that stand in for what a trained Graph Neural Network would learn. The code shows where a trained model plugs into the GA. Training a real GNN on traffic data is listed under future work.

---

## Getting started

**Requirements:** Python 3.8 or newer. No extra packages needed.

```bash
git clone https://github.com/btwitsamal/GNN-Route-Optimisation.git
cd GNN-Route-Optimisation
python gnn_route_optimisation.py
```

The demo finds a route from **Home** to **River_Dock** twice:

1. **Baseline GA** with fixed times and balanced weights (time 0.46, cost 0.44, comfort 0.1)
2. **GNN-enhanced GA** with traffic-aware times and comfort-focused weights (time 0.2, cost 0.1, comfort 0.7)

To try other journeys, change `start_location`, `end_location` or the weights at the bottom of the file.

---

## Example output

The GA is random, so results vary slightly between runs.

```
BASELINE GROUP EXECUTION (STATIC TIME)
Generation 0: Best fitness = 0.515
Generation 20: Best fitness = 0.344
Generation 40: Best fitness = 0.344

 BEST ROUTE FOUND BY BASELINE GA
Route: Home -> Bus_Station (Bus) -> Train_Station (Bus) -> City_Center (Bus) -> University (Bus) -> Hospital (Bus) -> Park_Entrance (Bus) -> River_Dock (Bus)
Cost: £14.41, Time: 143.00min, Comfort: 3.9/10, Fitness: 0.344

INDIVIDUAL REPORT: GNN-ENHANCED EXECUTION (DYNAMIC TIME)
Generation 0: Best fitness = 0.377
Generation 20: Best fitness = 0.278
Generation 40: Best fitness = 0.278

BEST ROUTE FOUND BY GNN-ENHANCED GA (h_GNN)
Route: Home -> Bus_Station (Taxi) -> Train_Station (Taxi) -> City_Center (Taxi) -> University (Taxi) -> Hospital (Taxi) -> Park_Entrance (Taxi) -> River_Dock (Taxi)
Cost: £113.50, Time: 112.32min, Comfort: 8.7/10, Fitness: 0.278
```

With balanced weights the GA picks a cheap all-bus route. When comfort is the priority it switches to taxis, which cost more but are faster and far more comfortable.

---

## Project structure

```
GNN-Route-Optimisation/
├── gnn_route_optimisation.py   # City graph, baseline GA, GNN-enhanced GA and demo
└── README.md
```

---

## Future work

- Replace the simulated estimator with a **trained GNN** (e.g. PyTorch Geometric) using real traffic data
- Move from weighted-sum fitness to **Pareto-based optimisation (NSGA-II)**, so users don't have to choose weights by hand

---

## Credits

- The baseline GA (`Route`, `RouteFinderGA`) was developed as group coursework.
- The overall GA and GNN structure was based on material from [JTexpo](https://jtexpo.github.io/).
- Google Gemini was used for help with some classes and code structure.

---

## Author

**Amal Thonniparambil Madhusudhanan**
BSc (Hons) Computer Science with Artificial Intelligence, University of Greenwich
[LinkedIn](https://linkedin.com/in/amal-tm) · [GitHub](https://github.com/btwitsamal)
