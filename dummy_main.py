import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_bode_tests(df: pd.DataFrame,
                    ax_mag: plt.Axes, 
                    ax_phase: plt.Axes,
                    title: str = "Bode Plot") -> tuple[plt.Axes, plt.Axes]:
    """
    Compute and plot Bode magnitude and phase responses for multiple test signals.
    
    Computes the FFT of each column (test signal), removes DC offset, and calculates
    magnitude (in dB) and phase (in degrees). Then overlays individual test responses
    with statistical bands (5th-95th percentile) and mean lines on a log-frequency scale.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each column is a time-domain signal and the index is time (seconds).
        Signals should be uniformly sampled.
    ax_mag : plt.Axes
        Matplotlib Axes object for the magnitude subplot (bottom).
    ax_phase : plt.Axes
        Matplotlib Axes object for the phase subplot (top).
    title : str, optional
        Title for the figure. Default is "Bode Plot".
    
    Returns
    -------
    tuple[plt.Axes, plt.Axes]
        Tuple of the magnitude and phase axes objects.
    
    Notes
    -----
    - DC component (0 Hz) is removed by subtracting the signal mean before FFT.
    - Magnitude is computed as 20 * log10(|Y|).
    - Phase is unwrapped to handle phase discontinuities and converted to degrees.
    - Plots use a semi-log X-axis (logarithmic frequency scale).
    """
    
    # Extract sampling interval from the time index (assuming uniform sampling)
    dt = np.diff(df.index.to_numpy())[0]   # Sampling period in seconds
    N  = len(df)                             # Number of samples per signal
    
    mags   = {}   # Dictionary: col → magnitude array (dB)
    phases = {}   # Dictionary: col → phase array (degrees)
    
    # ── Compute FFT per test signal ─────────────────────────────────────────────
    for col in df.columns:
        y     = df[col].to_numpy()          # Get column as numpy array
        Y     = np.fft.rfft(y - y.mean())   # Remove DC offset, compute real FFT
        freqs = np.fft.rfftfreq(N, d=dt)    # Compute frequency bins (Hz)
        
        # Drop DC bin (f=0) since log scale cannot display 0 dB/phase at f=0
        Y, freqs = Y[1:], freqs[1:]
        
        mags[col]   = 20 * np.log10(np.abs(Y))    # Magnitude in dB
        phases[col] = np.unwrap(np.angle(Y)) * 180 / np.pi  # Unwrapped phase in degrees
    
    # ── Stack into DataFrames for easy quantile/mean computation across tests ───
    df_mag   = pd.DataFrame(mags,   index=freqs)
    df_phase = pd.DataFrame(phases, index=freqs)
    
    # ── Plot statistics for magnitude and phase ────────────────────────────────
    for df_plot, ax, ylabel in [
        (df_mag,   ax_mag,   "Magnitude (dB)"),
        (df_phase, ax_phase, "Phase (degrees)")
    ]:
        # Compute mean response across all tests at each frequency
        mean = df_plot.mean(axis=1)
        
        # Compute 5th and 95th percentile bands for statistical coverage
        q05  = df_plot.quantile(0.05, axis=1)
        q95  = df_plot.quantile(0.95, axis=1)
        
        # Plot individual test responses (semi-transparent)
        for i, col in enumerate(df_plot.columns):
            ax.semilogx(freqs, df_plot[col], color="steelblue", alpha=0.3,
                        linewidth=0.8, label="Individual tests" if i == 0 else "")
        
        # Plot statistical bands and mean line
        ax.fill_between(freqs, q05, q95, color="steelblue", alpha=0.15,
                        label="5th–95th percentile band")
        ax.semilogx(freqs, mean, color="navy",       linewidth=2,   label="Mean response")
        ax.semilogx(freqs, q05,  color="darkorange", linewidth=1.2, linestyle="--", label="Lower bound (5th)")
        ax.semilogx(freqs, q95,  color="darkorange", linewidth=1.2, linestyle="--", label="Upper bound (95th)")
        
        # Configure axis labels and appearance
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, which="both", alpha=0.3)

    ax_mag.set_title(title)

    return ax_mag, ax_phase


# ── Example usage: generate dummy signals and plot Bode response ──────────────
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    fs = 100          # Sampling frequency: 100 Hz
    t  = np.arange(0, 10, 1 / fs)  # Time vector: 0 to 10 seconds
    
    # Generate 10 test signals, each containing:
    # - A sinusoid at 2 Hz (coherent across tests)
    # - A sinusoid at 10 Hz with half the amplitude
    # - Additive Gaussian noise
    df = pd.DataFrame(
        {f"test{i+1}": (np.sin(2 * np.pi * 2 * t) +
                        0.5 * np.sin(2 * np.pi * 10 * t) +
                        0.1 * np.random.randn(len(t)))
         for i in range(10)},
        index=t
    )
    
    # Create figure with two subplots: magnitude (bottom), phase (top)
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    plot_bode_tests(df, ax_mag=ax_mag, ax_phase=ax_phase, title="Power — Bode Plot")
    
    plt.tight_layout()
    plt.show()

    # plot a sibe wave int his file
    