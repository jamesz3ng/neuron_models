import matplotlib.pyplot as plt
import numpy as np

# Izhikevich neuron model parameters
a = 0.02
b = 0.2
c = -65.0
d = 8.0

T_MAX = 100  # Total simulation time in ms
dt = 0.5  # Time step in ms
time = np.arange(0, T_MAX, dt)
steps = len(time)

# Initial conditions
v = -65.0  # Initial membrane potential (mV)
u = b * v  # Initial recovery variable
I_input = 10.0  # Constant input current (pA)

# Data storage
v_trace = np.zeros(steps)
u_trace = np.zeros(steps)

# Simulation loop
for i in range(steps):
    v_trace[i] = v
    u_trace[i] = u

    # Check for spike and reset
    if v >= 30:
        v = c
        u = u + d

    # Update equations
    dv = 0.04 * v**2 + 5 * v + 140 - u + I_input
    du_dt = a * (b * v - u)

    # Euler integration
    v = v + dt * dv
    u = u + dt * du_dt


# Plotting
plt.figure(figsize=(10, 4))
plt.plot(time, v_trace, label="Membrane Potential (v)")
plt.title("Izhikevich Neuron Single Spike (Regular Spiking)")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.axhline(30, color="r", linestyle="--", linewidth=0.5, label="Spike Threshold")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.ylim(-80, 40)
plt.show()
