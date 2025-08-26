import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from matplotlib.animation import FuncAnimation, PillowWriter

class GrangerCausalityDemo:
    def __init__(self, n=420, lag=3, window=120, maxlag=5, seed=42):
        self.n = n
        self.lag = lag
        self.window = window
        self.maxlag = maxlag
        self.seed = seed
        np.random.seed(seed)
        self.data = None
        self.p_xy = []
        self.p_yx = []
        self.frames_idx = []

    # -------------------------------
    # 1) Simulate two time series
    # -------------------------------
    def simulate_data(self):
        x = np.zeros(self.n)
        eps_x = np.random.normal(scale=1.0, size=self.n)
        phi_x = 0.3
        for t in range(1, self.n):
            x[t] = phi_x * x[t-1] + eps_x[t]

        y = np.zeros(self.n)
        eps_y = np.random.normal(scale=1.0, size=self.n)
        phi_y = 0.4
        beta = 0.6
        for t in range(1, self.n):
            cause_term = beta * x[t-self.lag] if t - self.lag >= 0 else 0.0
            y[t] = phi_y * y[t-1] + cause_term + eps_y[t]

        self.data = pd.DataFrame({"X": x, "Y": y})

    # -------------------------------
    # 2) Rolling Granger causality
    # -------------------------------
    def _min_pvalue_granger(self, df_win, cause, effect):
        try:
            res = grangercausalitytests(df_win[[effect, cause]], maxlag=self.maxlag)

            pvals = [out[0]["ssr_ftest"][1] for _, out in res.items()]
            return float(np.min(pvals))
        except Exception:
            return np.nan

    def compute_rolling_pvalues(self):
        for t in range(self.window, self.n):
            df_win = self.data.iloc[t-self.window:t]
            self.p_xy.append(self._min_pvalue_granger(df_win, cause="X", effect="Y"))
            self.p_yx.append(self._min_pvalue_granger(df_win, cause="Y", effect="X"))
            self.frames_idx.append(t)
        self.p_xy = np.array(self.p_xy)
        self.p_yx = np.array(self.p_yx)
        self.frames_idx = np.array(self.frames_idx)

    # -------------------------------
    # 3) Animate and save GIF
    # -------------------------------
    def animate(self, gif_path="granger_demo.gif"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        line_x, = ax.plot([], [], label="X (driver)")
        line_y, = ax.plot([], [], label="Y (response)")
        text_box = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", ha="left")
        ax.set_xlim(0, self.n)
        ax.set_title("Granger Causality Demo")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")

        def init():
            line_x.set_data([], [])
            line_y.set_data([], [])
            text_box.set_text("")
            return line_x, line_y, text_box

        def update(frame_i):
            t = self.frames_idx[frame_i]
            line_x.set_data(np.arange(0, t+1), self.data["X"][:t+1])
            line_y.set_data(np.arange(0, t+1), self.data["Y"][:t+1])
            y_min = min(self.data["X"][:t+1].min(), self.data["Y"][:t+1].min())
            y_max = max(self.data["X"][:t+1].max(), self.data["Y"][:t+1].max())
            pad = 0.1 * (y_max - y_min + 1e-6)
            ax.set_ylim(y_min - pad, y_max + pad)
            txt = (
                f"Window: {self.window} (max lag = {self.maxlag})\n"
                f"p-value X → Y: {self.p_xy[frame_i]:.3g}\n"
                f"p-value Y → X: {self.p_yx[frame_i]:.3g}\n"
                f"(Lower p = stronger predictive link)"
            )
            text_box.set_text(txt)
            return line_x, line_y, text_box

        anim = FuncAnimation(fig, update, frames=len(self.frames_idx),
                             init_func=init, blit=True, interval=40)
        anim.save(gif_path, writer=PillowWriter(fps=25))
        print(f"GIF saved as {gif_path}")


# -------------------------------
# Run demo
# -------------------------------
demo = GrangerCausalityDemo()
demo.simulate_data()
demo.compute_rolling_pvalues()
demo.animate("𝗚𝗿𝗮𝗻𝗴𝗲𝗿 𝗖𝗮𝘂𝘀𝗮𝗹𝗶𝘁𝘆.gif")

