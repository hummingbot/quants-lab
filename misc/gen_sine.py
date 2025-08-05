import math
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

def main():
    # Parameters
    period1 = 10    # Initial period
    period2 = 50    # Final period
    N_samples = 2000  # Total number of samples to generate
    fs = 1000        # Sampling rate in Hz (samples per second)

    # Initialize the generator
    sine_generator = gen_sine(period1, period2)

    # Generate samples
    samples = [next(sine_generator) for _ in range(N_samples)]

    # Create time axis
    dt = 1 / fs  # Time between samples
    time = [i * dt for i in range(N_samples)]

    # Create the plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time,
        y=samples,
        mode='lines',
        name='Hyperbolic Chirp'
    ))

    # Update layout for better visualization
    fig.update_layout(
        title='Hyperbolic Chirp Signal',
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
