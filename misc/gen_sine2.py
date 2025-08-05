import math
from scipy.signal import butter
from collections import deque
import plotly.graph_objects as go
from nexus.charting import show_plotly_figure

def gen_sine(period1: int, period2: int):
    """
    Generates a hyperbolic chirp signal with amplitude 1.0 and a wave period
    changing linearly from period1 to period2.

    Args:
        period1 (int): Initial period of the sine wave.
        period2 (int): Final period of the sine wave.

    Yields:
        float: Next sample of the chirp signal.
    """
    phase = 0.0
    n = 0
    N = 1000  # Total number of samples over which the period changes
    while True:
        if n < N:
            # Linearly interpolate the period
            period = period1 + (period2 - period1) * n / N
        else:
            # After N samples, keep the period at period2
            period = period2
        sample = math.sin(phase)
        yield sample
        # Update phase
        phase += 2 * math.pi / period
        n += 1

def bandpass_filter(sample_generator, TimePeriod: float, Delta: float, fs: float, order: int = 4):
    """
    Applies a Butterworth bandpass filter to the incoming samples.

    Args:
        sample_generator (generator): Generator that yields input samples.
        TimePeriod (float): The central period to pass (in seconds).
        Delta (float): Determines the filter width (0.05 .. 1).
        fs (float): Sampling rate in Hz.
        order (int): Order of the Butterworth filter.

    Yields:
        float: Next filtered sample.
    """
    # Convert TimePeriod to frequency
    f_center = 1.0 / TimePeriod

    # Calculate bandwidth based on Delta
    # At Delta = 0.1, we want to pass frequencies within 30% range
    # Delta corresponds to the relative bandwidth
    # Let's define the passband as [f_center*(1 - Delta), f_center*(1 + Delta)]
    low = f_center * (1 - Delta)
    high = f_center * (1 + Delta)

    # Ensure frequencies are within valid range
    nyquist = 0.5 * fs
    low = max(low, 0.001)  # Avoid zero frequency
    high = min(high, nyquist - 0.001)  # Avoid exceeding Nyquist

    # Design Butterworth bandpass filter
    b, a = butter(N=order, Wn=[low / nyquist, high / nyquist], btype='band')

    # Initialize filter state (unused but kept for clarity)
    _ = [0.0] * max(len(a), len(b))  # Filter initial state

    # Create deque for previous inputs and outputs
    x_history = deque([0.0] * len(b), maxlen=len(b))
    y_history = deque([0.0] * len(a), maxlen=len(a))

    for x in sample_generator:
        # Update input history
        x_history.appendleft(x)

        # Compute the new output
        y = 0.0
        for i in range(len(b)):
            y += b[i] * x_history[i]
        for i in range(1, len(a)):
            y -= a[i] * y_history[i-1]

        # Normalize by a[0] if it's not 1
        if a[0] != 1.0:
            y /= a[0]

        # Update output history
        y_history.appendleft(y)

        yield y

def main():
    # Parameters for the chirp signal
    period1 = 10    # Initial period
    period2 = 50    # Final period
    N_samples = 2000  # Total number of samples to generate
    fs = 1000        # Sampling rate in Hz (samples per second)

    # Parameters for the bandpass filter
    TimePeriod = 30.0  # Target period to pass (in seconds)
    Delta = 0.1        # Filter width parameter (0.05 .. 1)

    # Initialize the sine generator
    sine_generator = gen_sine(period1, period2)

    # Apply the bandpass filter
    filtered_generator = bandpass_filter(sine_generator, TimePeriod, Delta, fs, order=4)

    # Generate samples
    original_samples = []
    filtered_samples = []
    for _ in range(N_samples):
        original = next(sine_generator)
        original_samples.append(original)
    # Reset the generator to apply filter again
    sine_generator = gen_sine(period1, period2)
    filtered_generator = bandpass_filter(sine_generator, TimePeriod, Delta, fs, order=4)
    for _ in range(N_samples):
        filtered = next(filtered_generator)
        filtered_samples.append(filtered)

    # Create time axis
    dt = 1 / fs  # Time between samples
    time = [i * dt for i in range(N_samples)]

    # Create the plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time,
        y=original_samples,
        mode='lines',
        name='Original Chirp',
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=time,
        y=filtered_samples,
        mode='lines',
        name='Filtered Signal',
        opacity=0.7
    ))

    # Update layout for better visualization
    fig.update_layout(
        title='Original and Bandpass Filtered Chirp Signal',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        template='plotly_dark',
        width=1000,
        height=600
    )

    # Show the plot
    show_plotly_figure(fig) # WSL workaround, instead of fig.show()

if __name__ == "__main__":
    main()
