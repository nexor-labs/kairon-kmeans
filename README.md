# KAIRON K-Means

A small educational implementation of the **K-Means clustering algorithm from scratch** using NumPy. Scikit-learn is used only to generate a demo dataset; the clustering algorithm itself is implemented in this project.

## Features

- K-Means implemented with NumPy
- Configurable number of clusters (`k`)
- Reproducible centroid initialization with a random seed
- Iteration limit and convergence tolerance
- Basic input validation
- Empty-cluster handling
- Cluster visualization with initial and final centroids
- Command-line options for common settings

## Preview

Run the project to generate `images/clustering-result.png`.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- scikit-learn (only for generating sample data)

## Installation

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**
```bat
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python kmeans.py
```

Choose a different number of clusters:

```bash
python kmeans.py --k 4
```

Change the iteration limit and random seed:

```bash
python kmeans.py --k 3 --max-iters 200 --seed 7
```

Save the plot without opening a window:

```bash
python kmeans.py --no-show
```

## How it works

1. Select `k` initial centroids from the dataset.
2. Assign each sample to its nearest centroid using Euclidean distance.
3. Update each centroid to the mean of the samples assigned to its cluster.
4. Repeat until the centroids stop moving beyond the tolerance or the iteration limit is reached.
5. Calculate inertia, the sum of squared distances from each sample to its assigned centroid.

## Project structure

```text
kairon-kmeans/
├── kmeans.py
├── requirements.txt
├── README.md
├── .gitignore
└── images/
    └── .gitkeep
```

## Limitations

- The included dataset is synthetic and two-dimensional for easy visualization.
- K-Means can converge to different results depending on initialization.
- Feature scaling may be important for real-world datasets.
- This is an educational implementation, not a replacement for a thoroughly optimized production library.

## License

MIT — see `LICENSE`.
