import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.interpolate import RegularGridInterpolator


def farfield_from_cst_reader(folder, name, mode='discrete'):
    def n_theta(data):
        s = 0
        theta_0 = data[1][0]
        while data[1][s] == theta_0:
            s += 1
        return s
    temp_path = folder / name
    data = np.loadtxt(temp_path, delimiter=None, skiprows=2).transpose()
    batches = data.shape[0]
    axe1 = data.shape[1]
    data = data.reshape(batches, axe1//n_theta(data), -1)
    data = np.concatenate((data, data[:, 0, :][:, None, :]), axis=1)
    theta = data[0, 0, :]
    phi = data[1, :, 0]
    if mode == 'discrete':
        return phi, theta, data[2:, :, :] #выводит массивы phi, theta и массив массивов [phi, theta]
    if mode == 'interpolated':
        phi[-1] = 360
        interpolated = RegularGridInterpolator((phi, theta),
            (data[2:, :, :]).transpose(1, 2, 0),
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
    return interpolated



class RadiationPattern3DPlotter:
    def __init__(self, width=1000, heigh=700):
        self.colors = [
            "#4B2AD5",
            "#4754DF",
            "#4097E6",
            "#3BCFEA",
            "#43F02F",
            "#86F12D",
            "#F0F22A",
            "#F8B42B",
            "#F67A2A",
            "#F2382E",
        ]
        self.width = width
        self.heigh = heigh
        self.title = None

    def set_title(self, title):
        self.title = title
        return self

    def plot(self, phi, theta, F, mode='linear', threshold=-50, renderer='browser'):
        pio.renderers.default = renderer
        F_colorbar = np.copy(F)
        if mode == 'logarithmic':
            threshold_liniar = 10 ** (threshold / 20)
            F_colorbar = 10*np.log10(np.copy(F))
            F_colorbar[F_colorbar < threshold] = threshold
            F /= threshold_liniar
            F[F < 1] = 1.01
            F = 10 * np.log10(F)
        theta, phi = np.radians(theta), np.radians(phi)

        # Пример: создаем сферическую сетку (theta, phi)
        THETA, PHI = np.meshgrid(theta, phi)

        R = F  # Значения ДН в сферических координатах

        # Переводим в декартовы координаты для Plotly
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)

        # Строим поверхность. Цвет (surfacecolor) задается значениями ДН

        fig = go.Figure(go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=F_colorbar,
            colorscale=self.colors,
            colorbar=dict(
                title=dict(
                    text="Directivity (dBi)",
                    side="right",
                    font=dict(size=22)
                ),
                tickfont=dict(size=18)
            )
        ))

        fig.update_layout(
            width=self.width,
            height=self.heigh,
            title=dict(
                text=self.title,
                x=0.5,
                xanchor="center"
            ),
            font=dict(size=20),
            scene=dict(
                aspectmode='data',

                # убрать подписи осей, деления и сами оси
                xaxis=dict(
                    visible=False
                ),
                yaxis=dict(
                    visible=False
                ),
                zaxis=dict(
                    visible=False
                ),
            )
        )
        fig.show()

