import numpy as np

import time
import math

# matplotlib.use('qtagg')
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def f(x):
    return x**3 - 3*x + 2


def plot_expression(expr_str, a, b, plot_file):
    n = 100

    x_vals = np.linspace(a, b, n + 1)

    # provide a restricted "safe" namespace with numpy/math helpers
    safe = {
        'np': np,
        'numpy': np,
        'math': math,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'pi': math.pi,
        'e': math.e,
    }

    # Try vectorized evaluation first (x is the whole numpy array)
    try:
        local = {'x': x_vals}
        y_vals = eval(expr_str, {"__builtins__": {}}, {**safe, **local})
        y_vals = np.array(y_vals)
        # if shapes don't match, force fallback
        if y_vals.shape != x_vals.shape:
            raise ValueError("Vectorized eval did not return expected shape")
    except Exception:
        # Fallback: evaluate expression for each scalar x
        y_list = []
        for x in x_vals:
            try:
                y = eval(expr_str, {"__builtins__": {}}, {**safe, 'x': x})
            except Exception:
                y = np.nan
            y_list.append(y)
        y_vals = np.array(y_list)

    fig = plt.figure()
    plt.plot(x_vals, y_vals)
    fig.savefig(plot_file)
    plt.close(fig)

# IMAGE: array/tenzor of 1920x1080x3

# A = 1, 2 | 3, 4; B = 1, -1 | 1, 1

# multiply to same-sized square matrix
def mat_mult(A, B):
    n = len(A)
    res = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(sum([A[i][k] * B[k][j] for k in range(n)]))
        res.append(row)
    return res


if __name__ == "__main__":

    n = 50

    # # generate random matrix with integers
    # A = np.random.randint(0, 10, (n, n))
    # B = np.random.randint(0, 10, (n, n))
    #
    # b = np.random.randint(0, 10, n)
    #
    # # solve linear system Ax = b
    # x = np.linalg.solve(A, b)
    #
    # # calculate norm / length of vector
    # x_len = np.linalg.norm(x)

    # A = [[1, 2], [3, 4]]
    # B = [[1, -1], [1, 1]]
    # A = np.array(A)
    # B = np.array(B)

    # multiply matrices with hand-made function and numpy and compare performance

    # start = time.time()
    # C = mat_mult(A, B)
    # end = time.time()
    # plain_time = end - start
    #
    # start = time.time()
    # C_numpy = A @ B
    # end = time.time()
    # numpy_time = end - start
    #
    # print(f"Plain time: {plain_time}, Numpy time: {numpy_time}, Speedup: {plain_time/numpy_time}")
    #
    # C = np.array(C)
    # # print(C)
    # # print(C_numpy)

    a = -2
    b = 2
    n = 100

    x_vals = np.linspace(a, b, n+1)
    y_vals = f(x_vals)

    # plain python version
    # x_vals = []
    # y_vals = []
    # for i in range(n+1):
    #     x_vals.append(a + (b-a)/n * i)
    #     y_vals.append(f(x_vals[-1]))


    # print(x_vals, y_vals)

    plt.plot(x_vals, y_vals)

    # save plot to file
    res_file = "static/plot.png"
    plt.savefig(res_file)

    # plt.show()