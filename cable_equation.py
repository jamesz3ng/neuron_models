import matplotlib.pyplot as plt
import numpy as np

L = 1000.0
T = 40.0
dx = 40.0
dt = 0.05

n_x = int(L / dx)
n_t = int(T / dt)

tau = 10.0
lam = 200.0

alpha = (lam**2 * dt) / (dx**2 * tau)
print(alpha)
beta = dt / tau


v_matrix = np.zeros((n_t, n_x))

# voltage high at x=0 for the first 5ms

input_steps = int(5.0 / dt)
v_matrix[:input_steps, 0] = 20.0

for n in range(0, n_t - 1):
    for i in range(1, n_x - 1):
        v_left = v_matrix[n, i - 1]
        v_center = v_matrix[n, i]
        v_right = v_matrix[n, i + 1]

        second_derivative = v_right - 2 * v_center + v_left
        diffusion = alpha * second_derivative

        decay = beta * v_center

        v_matrix[n + 1, i] = v_center + diffusion - decay

    # right boundary condition where x = L
    v_matrix[n + 1, -1] = v_matrix[n + 1, -2]
    # periodic
    # constant voltage

    # apply the left boundary condition i.e the stimulus
    if n < input_steps - 1:
        v_matrix[n + 1, 0] = 20.0

plt.figure(figsize=(10, 6))

plt.imshow(v_matrix, aspect="auto", cmap="viridis", origin="lower", extent=[0, L, 0, T])
plt.colorbar(label="Voltage")
plt.xlabel("Distance along the Neuron")
plt.ylabel("Time")
plt.show()
