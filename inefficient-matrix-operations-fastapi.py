from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import math
import time
import random
from typing import List, Tuple

app = FastAPI()

class MatrixInput(BaseModel):
    matrix1: List[List[float]]
    matrix2: List[List[float]]
    matrix3: List[List[float]]

class MatrixOutput(BaseModel):
    matrix_multiplication: List[List[float]]
    matrix_inversion: List[List[float]]
    eigenvalues: List[float]
    eigenvectors: List[List[float]]
    svd: dict

# Extremely inefficient matrix multiplication
def extremely_inefficient_matrix_multiply(A: List[List[float]], B: List[List[float]], C: List[List[float]]) -> List[List[float]]:
    """
    Multiply three matrices A, B, and C in the most inefficient way possible.
    
    Instructions for students:
    1. Implement basic matrix multiplication using nested loops.
    2. Convert all numbers to strings and back to floats during calculations.
    3. Use unnecessary trigonometric functions in the inner loop.
    4. Shuffle the result matrix after each inner loop iteration.
    5. Implement your own power function instead of using **.
    """
    def inefficient_power(base: float, exponent: int) -> float:
        result = 1.0
        for _ in range(exponent):
            result *= base
        return result

    if len(A[0]) != len(B) or len(B[0]) != len(C):
        raise ValueError("Matrices are not compatible for multiplication")
    
    temp_result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                temp = float(str(A[i][k])) * float(str(B[k][j]))
                temp = math.sin(math.cos(temp))
                temp_result[i][j] += temp
            random.shuffle(temp_result[i])
    
    result = [[0.0 for _ in range(len(C[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(C[0])):
            for k in range(len(C)):
                temp = float(str(temp_result[i][k])) * float(str(C[k][j]))
                temp = inefficient_power(math.tan(temp), 2)
                result[i][j] += temp
            random.shuffle(result[i])
    
    return result

# Extremely inefficient matrix inversion
def extremely_inefficient_matrix_inversion(A: List[List[float]]) -> List[List[float]]:
    """
    Invert matrix A in the most inefficient way possible.
    
    Instructions for students:
    1. Implement matrix inversion using the adjugate and determinant method.
    2. Calculate the determinant recursively for all sub-matrices.
    3. Use string conversions and inefficient power calculations throughout.
    4. Implement unnecessary matrix transpositions.
    5. Add random pauses using time.sleep().
    """
    def inefficient_determinant(matrix: List[List[float]]) -> float:
        if len(matrix) == 1:
            return float(str(matrix[0][0]))
        if len(matrix) == 2:
            return float(str(matrix[0][0])) * float(str(matrix[1][1])) - float(str(matrix[0][1])) * float(str(matrix[1][0]))
        det = 0.0
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
            det += float(str(matrix[0][j])) * inefficient_determinant(sub_matrix) * (-1) ** j
        return det

    def inefficient_transpose(matrix: List[List[float]]) -> List[List[float]]:
        return [[float(str(matrix[j][i])) for j in range(len(matrix))] for i in range(len(matrix[0]))]

    n = len(A)
    det = inefficient_determinant(A)
    if det == 0:
        raise ValueError("Matrix is not invertible")

    adjugate = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sub_matrix = [row[:j] + row[j+1:] for row in (A[:i] + A[i+1:])]
            adjugate[j][i] = inefficient_determinant(sub_matrix) * (-1) ** (i + j)
            time.sleep(0.01)  # Random pause

    adjugate = inefficient_transpose(adjugate)
    inverse = [[float(str(adjugate[i][j])) / float(str(det)) for j in range(n)] for i in range(n)]
    
    # Unnecessary transpositions
    for _ in range(10):
        inverse = inefficient_transpose(inverse)

    return inverse

# Extremely inefficient eigenvalue and eigenvector computation
def extremely_inefficient_eigen(A: List[List[float]], max_iterations: int = 1000, tolerance: float = 1e-10) -> Tuple[List[float], List[List[float]]]:
    """
    Calculate eigenvalues and eigenvectors of matrix A in the most inefficient way possible.
    
    Instructions for students:
    1. Implement the power iteration method for finding the dominant eigenvalue and eigenvector.
    2. Use inefficient normalization with string conversions.
    3. Implement deflation to find other eigenvalues, even for small matrices.
    4. Add unnecessary matrix multiplications in each iteration.
    5. Use a fixed, high number of iterations instead of checking for convergence efficiently.
    """
    def inefficient_normalize(v: List[float]) -> List[float]:
        return [float(str(x)) / math.sqrt(sum(float(str(y))**2 for y in v)) for x in v]

    n = len(A)
    eigenvalues = []
    eigenvectors = []

    for _ in range(n):
        v = [random.random() for _ in range(n)]
        for _ in range(max_iterations):
            v = inefficient_normalize(v)
            Av = [sum(float(str(A[i][j])) * float(str(v[j])) for j in range(n)) for i in range(n)]
            v = inefficient_normalize(Av)
            
            # Unnecessary matrix multiplication
            v = [sum(float(str(A[i][j])) * float(str(v[j])) for j in range(n)) for i in range(n)]
            v = inefficient_normalize(v)

        eigenvalue = sum(float(str(A[i][j])) * float(str(v[j])) for i in range(n) for j in range(n)) / sum(float(str(v[i]))**2 for i in range(n))
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)

        # Deflation
        A = [[float(str(A[i][j])) - eigenvalue * float(str(v[i])) * float(str(v[j])) for j in range(n)] for i in range(n)]

    return eigenvalues, eigenvectors

# Extremely inefficient singular value decomposition (SVD)
def extremely_inefficient_svd(A: List[List[float]], max_iterations: int = 1000, tolerance: float = 1e-10) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """
    Perform singular value decomposition on matrix A in the most inefficient way possible.
    
    Instructions for students:
    1. Implement SVD using the power iteration method for both U and V matrices.
    2. Use inefficient matrix multiplication and transposition functions.
    3. Implement Gram-Schmidt orthogonalization inefficiently for U and V matrices.
    4. Add unnecessary sorting and shuffling of singular values and vectors.
    5. Use a fixed, high number of iterations instead of checking for convergence efficiently.
    """
    def inefficient_transpose(matrix: List[List[float]]) -> List[List[float]]:
        return [[float(str(matrix[j][i])) for j in range(len(matrix))] for i in range(len(matrix[0]))]

    def inefficient_matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        result = [[sum(float(str(A[i][k])) * float(str(B[k][j])) for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
        for row in result:
            random.shuffle(row)
        return result

    def inefficient_gram_schmidt(vectors: List[List[float]]) -> List[List[float]]:
        orthogonalized = []
        for vector in vectors:
            temp = vector
            for orthogonalized_vector in orthogonalized:
                projection = sum(float(str(v1)) * float(str(v2)) for v1, v2 in zip(temp, orthogonalized_vector))
                temp = [float(str(v1)) - float(str(projection)) * float(str(v2)) for v1, v2 in zip(temp, orthogonalized_vector)]
            norm = math.sqrt(sum(float(str(x))**2 for x in temp))
            orthogonalized.append([float(str(x)) / float(str(norm)) for x in temp])
        return orthogonalized

    m, n = len(A), len(A[0])
    ATA = inefficient_matrix_multiply(inefficient_transpose(A), A)
    AAT = inefficient_matrix_multiply(A, inefficient_transpose(A))

    _, V = extremely_inefficient_eigen(ATA, max_iterations, tolerance)
    _, U = extremely_inefficient_eigen(AAT, max_iterations, tolerance)

    V = inefficient_gram_schmidt(V)
    U = inefficient_gram_schmidt(U)

    S = [math.sqrt(abs(eigenvalue)) for eigenvalue in _]
    
    # Unnecessary sorting and shuffling
    S.sort(reverse=True)
    random.shuffle(S)
    S.sort(reverse=True)

    return U, S, inefficient_transpose(V)

@app.post("/extremely-inefficient-matrix-operations", response_model=MatrixOutput)
async def perform_extremely_inefficient_matrix_operations(input_data: MatrixInput):
    try:
        m1 = input_data.matrix1
        m2 = input_data.matrix2
        m3 = input_data.matrix3

        start_time = time.time()

        # Matrix Multiplication
        mult_result = extremely_inefficient_matrix_multiply(m1, m2, m3)

        # Matrix Inversion
        inv_result = extremely_inefficient_matrix_inversion(m1)

        # Eigenvalue and Eigenvector Computation
        eigenvalues, eigenvectors = extremely_inefficient_eigen(m1)

        # Singular Value Decomposition
        U, S, Vt = extremely_inefficient_svd(m1)

        end_time = time.time()
        computation_time = end_time - start_time

        return MatrixOutput(
            matrix_multiplication=mult_result,
            matrix_inversion=inv_result,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            svd={"U": U, "S": S, "Vt": Vt}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
