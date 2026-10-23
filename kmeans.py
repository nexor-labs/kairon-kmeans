import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def kmeans(X, k, max_iters=100, tol=1e-4, random_state=42):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite numeric values.")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise ValueError("k must be an integer.")
    if not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and the number of samples.")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer.")
    if tol < 0:
        raise ValueError("tol must be non-negative.")

    rng = np.random.default_rng(random_state)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    history = [centers.copy()]

    for iteration in range(1, max_iters + 1):
        distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()

        for cluster_id in range(k):
            cluster_points = X[labels == cluster_id]

            if len(cluster_points) > 0:
                new_centers[cluster_id] = cluster_points.mean(axis=0)
            else:
                nearest_distances = distances.min(axis=1)
                candidate_indices = np.argsort(nearest_distances)[::-1]
                for candidate_index in candidate_indices:
                    if not any(np.array_equal(X[candidate_index], c) for c in new_centers):
                        new_centers[cluster_id] = X[candidate_index]
                        break

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        history.append(centers.copy())

        if shift <= tol:
            break

    final_distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    labels = np.argmin(final_distances, axis=1)
    inertia = float(np.sum((X - centers[labels]) ** 2))

    return centers, labels, iteration, inertia, history


def create_demo_data(random_state=123):
    X, _ = make_blobs(
        n_samples=300,
        centers=[(7, 7), (1, 5), (-2, 4)],
        cluster_std=1.0,
        random_state=random_state,
    )
    return X


def plot_clusters(X, labels, centers, history, output_path=None, show=True):
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=120)

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        s=36,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
    )

    initial_centers = history[0]
    ax.scatter(
        initial_centers[:, 0],
        initial_centers[:, 1],
        marker="X",
        s=180,
        c="#f59e0b",
        edgecolors="#111827",
        linewidths=1.0,
        label="Initial centroids",
        zorder=4,
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="*",
        s=260,
        c="#ef4444",
        edgecolors="white",
        linewidths=0.8,
        label="Final centroids",
        zorder=5,
    )

    ax.set_title("K-Means Clustering from Scratch", fontsize=16, pad=14)
    ax.set_xlabel("Feature 1", fontsize=11)
    ax.set_ylabel("Feature 2", fontsize=11)
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=True)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run a from-scratch K-Means clustering demo."
    )
    parser.add_argument("--k", type=int, default=3, help="Number of clusters (default: 3).")
    parser.add_argument("--max-iters", type=int, default=100, help="Maximum iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument(
        "--save",
        type=str,
        default="images/clustering-result.png",
        help="Path to save the plot.",
    )
    parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a window.")
    args = parser.parse_args()

    X = create_demo_data()
    centers, labels, iterations, inertia, history = kmeans(
        X,
        k=args.k,
        max_iters=args.max_iters,
        tol=args.tol,
        random_state=args.seed,
    )

    print("K-Means completed successfully")
    print(f"Samples: {len(X)}")
    print(f"Clusters: {args.k}")
    print(f"Iterations: {iterations}")
    print(f"Inertia: {inertia:.4f}")
    print("Final centroids:")
    print(centers)

    plot_clusters(
        X,
        labels,
        centers,
        history,
        output_path=args.save,
        show=not args.no_show,
    )
    print(f"Plot saved to: {args.save}")


if __name__ == "__main__":
    main()
