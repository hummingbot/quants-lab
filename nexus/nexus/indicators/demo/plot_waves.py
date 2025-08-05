import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nexus.indicators.utils import push, genSine, genSquare, genNoise
from nexus.charting import show_plotly_figure

# Simulation parameters
num_bars = 800  # Total number of bars
look_back = 100  # Look-back period
period1 = 3  # Starting period for sine/square wave
period2 = 60  # Ending period for sine/square wave

# Data lists to hold generated waveforms
sine_wave = np.zeros(num_bars-look_back)
square_wave = np.zeros(num_bars-look_back)
noise_wave = np.zeros(num_bars-look_back)

# Generate the series by looping over the bars
for nBar in range(num_bars):
    push(sine_wave, genSine(period1, period2, num_bars, look_back, nBar))
    push(square_wave, genSquare(period1, period2, num_bars, look_back, nBar))
    push(noise_wave, genNoise())

# Create subplots
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Sine Wave", "Square Wave", "Noise Wave"),
    shared_xaxes=True,
    vertical_spacing=0.05
)

# Add sine wave
fig.add_trace(
    go.Scatter(x=list(range(num_bars)), y=sine_wave, mode='lines', name='Sine Wave'),
      row=1, col=1
    )

# Add square wave
fig.add_trace(
    go.Scatter(x=list(range(num_bars)), y=square_wave, mode='lines', name='Square Wave'),
    row=2, col=1
    )

# Add noise wave
fig.add_trace(
    go.Scatter(x=list(range(num_bars)), y=noise_wave, mode='lines', name='Noise Wave'),
    row=3, col=1
)

# Update layout
fig.update_layout(
    height=900,
    width=1200,
    title_text="Waveform Generators Demo",
    showlegend=False
)

# Update axes labels
fig.update_xaxes(title_text="Sample Number", row=3, col=1)
fig.update_yaxes(title_text="Amplitude", row=1, col=1)
fig.update_yaxes(title_text="Amplitude", row=2, col=1)
fig.update_yaxes(title_text="Amplitude", row=3, col=1)

show_plotly_figure(fig) # WSL workaround, instead of fig.show()
