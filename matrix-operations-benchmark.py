import time
import requests
import numpy as np
from typing import List, Dict
import json

# Assuming the FastAPI server is running on localhost:8000
BASE_URL = "http://localhost:8000"

def generate_random_matrix(n: int) -> List[List[float]]:
    return np.random.rand(n, n).tolist()

def make_request(matrices: Dict[str, List[List[float]]]) -> Dict:
    response = requests.post(f"{BASE_URL}/extremely-inefficient-matrix-operations", json=matrices)
    return response.json()

def benchmark(matrix_sizes: List[int], num_runs: int = 10) -> Dict[str, Dict[str, float]]:
    results = {}
    
    for size in matrix_sizes:
        print(f"Benchmarking with matrix size {size}x{size}")
        matrices = {
            "matrix1": generate_random_matrix(size),
            "matrix2": generate_random_matrix(size),
            "matrix3": generate_random_matrix(size)
        }
        
        operation_times = {
            "matrix_multiplication": [],
            "matrix_inversion": [],
            "eigenvalues": [],
            "svd": []
        }
        
        for i in range(num_runs):
            print(f"  Run {i + 1}/{num_runs}")
            start_time = time.time()
            response = make_request(matrices)
            end_time = time.time()
            
            total_time = end_time - start_time
            operation_times["matrix_multiplication"].append(total_time)
            operation_times["matrix_inversion"].append(total_time)
            operation_times["eigenvalues"].append(total_time)
            operation_times["svd"].append(total_time)
        
        results[size] = {
            op: np.mean(times) for op, times in operation_times.items()
        }
        results[size]["total"] = np.mean([sum(times) for times in zip(*operation_times.values())])
    
    return results

def calculate_speedup(baseline_results: Dict[str, Dict[str, float]], 
                      optimized_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    speedup = {}
    for size in baseline_results.keys():
        speedup[size] = {}
        for operation in baseline_results[size].keys():
            baseline_time = baseline_results[size][operation]
            optimized_time = optimized_results[size][operation]
            speedup[size][operation] = baseline_time / optimized_time
    return speedup

def main():
    matrix_sizes = [10, 50, 100, 200, 500]  # Increased matrix sizes
    num_runs = 10  # Increased number of runs
    
    print("Running baseline benchmark...")
    baseline_results = benchmark(matrix_sizes, num_runs)
    
    print("\nBaseline results:")
    print(json.dumps(baseline_results, indent=2))
    
    input("\nPress Enter to run the optimized version benchmark...")
    
    print("\nRunning optimized version benchmark...")
    optimized_results = benchmark(matrix_sizes, num_runs)
    
    print("\nOptimized results:")
    print(json.dumps(optimized_results, indent=2))
    
    speedup = calculate_speedup(baseline_results, optimized_results)
    
    print("\nSpeedup (higher is better):")
    print(json.dumps(speedup, indent=2))

    # Print a summary of total execution times and speedup
    print("\nSummary:")
    print("Matrix Size | Baseline Total (s) | Optimized Total (s) | Total Speedup")
    print("---------------------------------------------------------------------------")
    for size in matrix_sizes:
        baseline_total = baseline_results[size]["total"]
        optimized_total = optimized_results[size]["total"]
        total_speedup = speedup[size]["total"]
        print(f"{size:11d} | {baseline_total:18.2f} | {optimized_total:19.2f} | {total_speedup:13.2f}")

if __name__ == "__main__":
    main()
