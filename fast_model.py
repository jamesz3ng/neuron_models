import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION ---
L_dendrite = 10.0
c = 25.0
T_duration = 0.5
dt = 0.001

# Calculate the exact delay in seconds
delay_seconds = L_dendrite / c

# Calculate how many "array indices" this delay corresponds to
# (Round to nearest integer)
delay_steps = int(round(delay_seconds / dt))

print(f"Delay is {delay_seconds} seconds ({delay_steps} time steps)")

# Time array
t = np.arange(0, T_duration, dt)


# generate input signal
def synaptic_input(time):
    peak_time = 0.05
    width = 0.01
    val = 100 * np.exp(-((time - peak_time) ** 2) / (2 * width**2))
    # remove floating point noise for cleaner zeroes
    return np.where(val < 1e-5, 0, val)


# Create the entire history of the input at once (Vectorized = FAST)
v_source = synaptic_input(t)
print(v_source)

# --- 3. CALCULATE SOMA VOLTAGE (The "Math" Step) ---
# We want: v_soma[i] = v_source[i - delay_steps]

# Initialize array with zeros
v_soma = np.zeros_like(v_source)
print(v_soma)
# If the simulation time index is greater than the delay,
# copy the value from the past.
# We use Python slicing for maximum speed (no for loops needed!)
if delay_steps < len(t):
    v_soma[delay_steps:] = v_source[:-delay_steps]

# --- 4. VISUALIZATION ---
plt.figure(figsize=(10, 6))
plt.plot(t, v_source, "b--", label="Input (Dendrite Tip)")
plt.plot(t, v_soma, "r-", lw=2, label="Output (Soma)")

plt.title(f"Exact Delay Model (Delay = {delay_seconds}s)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.grid(True)
plt.show()
