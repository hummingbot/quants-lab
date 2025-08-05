import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
from nexus.indicators.utils import genNoise         
from nexus.indicators.indicators import aci, fractal_dimension, hurst_exponent, mmi   
from nexus.indicators.ehlers import SuperSmoother
from nexus.charting import show_plotly_figure

num_bars     = 1500
look_back    = 500
trend_start  = int(look_back + 0.4*(num_bars-look_back))
time_period  = look_back // 2

buf          = deque(maxlen=look_back)           # rolling window
signal       = []                                # chronological list
aci_series   = []
fd_series    = []
hurst_series = []
hurst_series_smooth = []
mmi_series   = []

super_smoother_filter = SuperSmoother(cutoff=look_back/10)

for n in range(num_bars):
    val = 1 + 0.5*genNoise()
    if n > trend_start:
        val += 0.003*(n - trend_start)           # upward drift
    buf.append(val)

    if len(buf) == look_back:
        signal.append(val)
        aci_series.append(aci(np.asarray(buf), look_back, 50))
        fd_series.append(fractal_dimension(np.asarray(buf), look_back))
        hurst_series.append(hurst_exponent(np.asarray(buf), look_back))
        hurst_series_smooth.append(super_smoother_filter.update(hurst_series[-1]))
        mmi_series.append(mmi(np.asarray(buf), look_back))

# ------------------------ plotting ----------------------------------
fig = make_subplots(rows=5, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=("Signal", "Auto-Correlation Index (ACI)", "Fractal Dimension", "Hurst Exponent", "Market Meanness Index (MMI)"),)

# regime colouring: threshold ≈ 0.25 works well for the demo
thr = 0.25
side_idxs = [i for i, v in enumerate(aci_series) if v < thr]
trend_idxs = [i for i, v in enumerate(aci_series) if v >= thr]

fig.add_trace(go.Scatter(x=side_idxs,   y=np.take(signal, side_idxs),
                         mode='lines', line=dict(color='black'),
                         name='Sideways'), row=1, col=1)
fig.add_trace(go.Scatter(x=trend_idxs,  y=np.take(signal, trend_idxs),
                         mode='lines', line=dict(color='royalblue'),
                         name='Trending'), row=1, col=1)

fig.add_trace(go.Scatter(x=list(range(len(aci_series))),
                         y=aci_series, mode='lines',
                         line=dict(color='firebrick'),
                         name='ACI'), row=2, col=1)

fig.add_trace(go.Scatter(x=list(range(len(fd_series))),
                         y=fd_series, mode='lines',
                         line=dict(color='firebrick'),
                         name='Fractal Dimension'), row=3, col=1)

fig.add_trace(go.Scatter(x=list(range(len(hurst_series))),
                         y=hurst_series, mode='lines',
                         line=dict(color='firebrick'),
                         name='Hurst Exponent'), row=4, col=1)    
    
fig.add_trace(go.Scatter(x=list(range(len(hurst_series_smooth))),
                         y=hurst_series_smooth, mode='lines',
                         line=dict(color='orange'),
                         name='Hurst Exponent Smooth'), row=4, col=1)  

fig.add_trace(go.Scatter(x=list(range(len(mmi_series))),
                         y=mmi_series, mode='lines',
                         line=dict(color='firebrick'),
                         name='Market Meanness Index'), row=5, col=1)                 

fig.update_layout(height=900, width=1200,
                  title_text="Regime detection demo",
                  showlegend=False
                  )
show_plotly_figure(fig) # WSL workaround, instead of fig.show()
