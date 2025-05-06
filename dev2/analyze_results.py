import os
import re
import csv
import pstats
import matplotlib.pyplot as plt

LOG_DIR = "logs"
SUMMARY_CSV = "summary.csv"

def parse_cprofile(file_path):
    p = pstats.Stats(file_path)
    return p.total_tt

def parse_runtime_from_log(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    match = re.search(r'(\d+\.\d+)\s*seconds?', content, re.IGNORECASE)
    return float(match.group(1)) if match else None

def main():
    rows = []

    for fname in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, fname)

        if fname.startswith("seq_") and fname.endswith(".prof"):
            m = re.match(r"seq_n(\d+)_k(\d+)\.prof", fname)
            if m:
                n, k = map(int, m.groups())
                runtime = parse_cprofile(path)
                rows.append(["sequential", n, k, 1, 1, runtime])

        elif fname.startswith("mpi_") and fname.endswith(".log"):
            m = re.match(r"mpi_n(\d+)_k(\d+)_np(\d+)\.log", fname)
            if m:
                n, k, np = map(int, m.groups())
                runtime = parse_runtime_from_log(path)
                rows.append(["mpi", n, k, int(np), 1, runtime])

        elif fname.startswith("hybrid_") and fname.endswith(".log"):
            m = re.match(r"hybrid_n(\d+)_k(\d+)_np(\d+)_omp(\d+)\.log", fname)
            if m:
                n, k, np, omp = map(int, m.groups())
                runtime = parse_runtime_from_log(path)
                rows.append(["hybrid", n, k, np, omp, runtime])

    # Compute speedup/efficiency
    summary = []
    for r in rows:
        impl, n, k, np, omp, time = r
        if impl == "sequential":
            continue
        baseline = next((s[5] for s in rows if s[0] == "sequential" and s[1] == n and s[2] == k), None)
        if baseline:
            speedup = baseline / time if time else None
            efficiency = speedup / (np * omp)
            summary.append([impl, n, k, np, omp, time, speedup, efficiency])

    # Write summary.csv
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Implementation", "n", "k", "MPI Procs", "OMP Threads", "Runtime", "Speedup", "Efficiency"])
        writer.writerows(summary)

    print(f"Saved summary to {SUMMARY_CSV}")

    # Optional plot for one n,k
    plot_n, plot_k = 10000, 4
    filtered = [r for r in summary if r[1] == plot_n and r[2] == plot_k]
    plt.figure(figsize=(10, 6))
    for impl in set(r[0] for r in filtered):
        xs = [r[3]*r[4] for r in filtered if r[0] == impl]
        ys = [r[5] for r in filtered if r[0] == impl]
        plt.plot(xs, ys, marker='o', label=impl)
    plt.title(f"Runtime for n={plot_n}, k={plot_k}")
    plt.xlabel("Total Threads (MPI × OMP)")
    plt.ylabel("Runtime (s)")
    plt.grid(True)
    plt.legend()
    plt.savefig("plot.png")
    print("📊 Saved plot as plot.png")

if __name__ == "__main__":
    main()
