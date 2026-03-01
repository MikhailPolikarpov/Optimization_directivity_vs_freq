import plotly.io as pio #Хахахаха проверка связи
pio.renderers.default = "browser"
import numpy as np
import plotly.graph_objects as go

import plotly.io as pio

# открывать график в браузере
pio.renderers.default = "browser"
theta = np.linspace(-np.pi/2, np.pi/2, 100)
phi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

R = np.abs(np.cos(theta))**2

X = R * np.sin(theta) * np.cos(phi)
Y = R * np.sin(theta) * np.sin(phi)
Z = R * np.cos(theta)

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
fig.show()