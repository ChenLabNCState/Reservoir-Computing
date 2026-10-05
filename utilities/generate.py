import numpy as np
import random

#Generate random Sin-Square wave pulse
def generate_mixed_amplitude_sequence(
    total_points=150,
    segment_length_min=10,
    segment_length_max=10,
    sine_amplitude=1.0,
    square_amplitude=1.0,
    frequency=0.2,
    noise_level=0.0
):
    sequence = []
    labels = []
    current_points = 0

    while current_points < total_points:
        length = random.randint(segment_length_min, segment_length_max)
        if current_points + length > total_points:
            length = total_points - current_points

        pulse_type = random.choice(['sine', 'square'])
        t = np.arange(length)

        if pulse_type == 'sine':
            pulse = sine_amplitude * np.sin(2 * np.pi * frequency * t)
            label = 0
        else:
            pulse = square_amplitude * np.ones(length)
            label = 1

        if noise_level > 0:
            pulse += np.random.normal(0, noise_level, length)

        sequence.extend(pulse)
        labels.extend([label] * length)
        current_points += length

    return np.array(sequence), np.array(labels)

#Generate chaotic time series data based on Mackey Glass - Diffeq
def generate_mackey_glass(length, dt=0.1, tau=17, beta=0.2, gamma=0.1, n=10):
    delay_steps = int(tau / dt)
    history = np.ones(delay_steps) * 0.9
    x = history[-1]
    series = []
    for _ in range(length):
        x_tau = history[0]
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x
        x += dx * dt
        series.append(x)
        history = np.roll(history, -1)
        history[-1] = x
    series = np.array(series)
    return series / (series.max()) + 0.5



