import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURATION ---
L_dendrite = 10.0  # Length of dendrite (micrometers or arbitrary units)
c = 50.0  # Conduction velocity (slower in dendrites usually)
T_duration = 0.5  # Total simulation time

dx = 0.1
dt = 0.001

# CFL Condition Check
courant_number = c * dt / dx
print(courant_number)
if courant_number > 1:
    print("unstable")

x = np.arange(0, L_dendrite, dx)
t_steps = np.arange(0, T_duration, dt)
n_points = len(x)
n_steps = len(t_steps)

# Initialize Voltage
V = np.zeros(n_points)
soma_voltage_history = []  # To record what the soma sees


# input
def synaptic_input(time):
    # A burst happens early, at t=0.05
    peak_time = 0.05
    width = 0.01
    return 100 * np.exp(-((time - peak_time) ** 2) / (2 * width**2))


# --- 3. SIMULATION ---
V_history = []

for n in range(n_steps):
    current_time = n * dt

    # 1. Inject signal at the Dendrite Tip (x=0)
    V[0] = synaptic_input(current_time)

    # 2. Propagate (Upwind Scheme)
    V_new = V.copy()
    V_new[1:] = V[1:] - courant_number * (V[1:] - V[:-1])
    V = V_new

    # 3. Record History
    V_history.append(V.copy())

    # 4. Measure voltage at the Soma (The last point, x=L)
    soma_voltage_history.append(V[-1])

# --- 4. VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(hspace=0.4)

# Plot 1: The Wave moving along the dendrite
ax1.set_xlim(0, L_dendrite)
ax1.set_ylim(-10, 110)
ax1.set_title("Signal Traveling down Dendrite")
ax1.set_xlabel("Distance (x)")
ax1.set_ylabel("Voltage (mV)")
(line1,) = ax1.plot([], [], "b-", lw=2)
# Draw a red line to represent the Soma at the end
ax1.axvline(x=L_dendrite - 0.2, color="r", linestyle="--", label="Soma")
ax1.legend(loc="upper right")

# Plot 2: The Voltage AT THE SOMA over time
ax2.set_xlim(0, T_duration)
ax2.set_ylim(-10, 110)
ax2.set_title("Voltage Recorded at Soma (x=L)")
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Voltage (mV)")
(line2,) = ax2.plot([], [], "r-", lw=2)


def animate(i):
    # Update Wave
    line1.set_data(x, V_history[i])

    # Update Soma Trace (draw line from t=0 up to current time)
    current_t_data = t_steps[:i]
    current_v_data = soma_voltage_history[:i]
    line2.set_data(current_t_data, current_v_data)

    return line1, line2


anim = FuncAnimation(fig, animate, frames=range(0, n_steps, 10), interval=20, blit=True)

plt.show()
