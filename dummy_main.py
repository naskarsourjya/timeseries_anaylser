import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_bode_tests(df: pd.DataFrame,
                    ax_mag: plt.Axes, ax_phase: plt.Axes,
                    title: str = "Bode Plot") -> tuple[plt.Axes, plt.Axes]:
    """
    Plot Bode diagram (magnitude & phase) for all test columns in df,
    plus mean, 5th and 95th quantiles.

    Parameters
    ----------
    df      : pd.DataFrame
        DataFrame with time as index and one column per test (e.g. test1, test2, ...).
    ax_mag  : plt.Axes
        Axes to plot magnitude (dB) on.
    ax_phase: plt.Axes
        Axes to plot phase (degrees) on.
    title   : str
        Title added to ax_mag.

    Returns
    -------
    ax_mag, ax_phase : plt.Axes
    """
    dt = np.diff(df.index.to_numpy())[0]   # sampling interval from index
    N  = len(df)

    mags   = {}   # col → magnitude array (dB)
    phases = {}   # col → phase array (degrees)

    # ── Compute FFT per test ─────────────────────────────────────────────
    for col in df.columns:
        y     = df[col].to_numpy()
        Y     = np.fft.rfft(y - y.mean())       # remove DC offset
        freqs = np.fft.rfftfreq(N, d=dt)

        # drop DC bin (f=0) for log-scale
        Y, freqs = Y[1:], freqs[1:]

        mags[col]   = 20 * np.log10(np.abs(Y))
        phases[col] = np.unwrap(np.angle(Y)) * 180 / np.pi

    # ── Stack into DataFrames for easy quantile/mean computation ─────────
    df_mag   = pd.DataFrame(mags,   index=freqs)
    df_phase = pd.DataFrame(phases, index=freqs)

    # ── Statistics ───────────────────────────────────────────────────────
    for df_plot, ax, ylabel in [
        (df_mag,   ax_mag,   "Magnitude (dB)"),
        (df_phase, ax_phase, "Phase (degrees)")
    ]:
        mean = df_plot.mean(axis=1)
        q05  = df_plot.quantile(0.05, axis=1)
        q95  = df_plot.quantile(0.95, axis=1)

        # Individual tests
        for i, col in enumerate(df_plot.columns):
            ax.semilogx(freqs, df_plot[col], color="steelblue", alpha=0.3,
                        linewidth=0.8, label="Individual tests" if i == 0 else "")

        # Quantile band
        ax.fill_between(freqs, q05, q95, color="steelblue", alpha=0.15,
                        label="5th–95th percentile")

        # Mean and quantile lines
        ax.semilogx(freqs, mean, color="navy",       linewidth=2,   label="Mean")
        ax.semilogx(freqs, q05,  color="darkorange", linewidth=1.2, linestyle="--", label="5th quantile")
        ax.semilogx(freqs, q95,  color="darkorange", linewidth=1.2, linestyle="--", label="95th quantile")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    ax_mag.set_title(title)

    return ax_mag, ax_phase


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    fs = 100
    t  = np.arange(0, 10, 1 / fs)

    df = pd.DataFrame(
        {f"test{i+1}": (np.sin(2 * np.pi * 2 * t) +
                        0.5 * np.sin(2 * np.pi * 10 * t) +
                        0.1 * np.random.randn(len(t)))
         for i in range(10)},
        index=t
    )

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plot_bode_tests(df, ax_mag=ax0, ax_phase=ax1, title="Power — Bode Plot")
    plt.tight_layout()
    plt.show()