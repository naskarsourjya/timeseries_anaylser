import numpy as np
import matplotlib.pyplot as plt

def plot_bode(signal: np.ndarray, freq: np.ndarray):
    """
    Generates and plots the Bode diagram given a signal and corresponding frequency array.

    Args:
        signal (np.ndarray): The complex or magnitude response of the system/signal 
                              H(j*w).
        freq (np.ndarray): The corresponding frequency domain axis (Hz).
    """
    if len(signal) != len(freq):
        print("Error: Signal and Frequency arrays must have the same length.")
        return

    plt.figure()
    
    # Magnitude Plot (20 * log10(|H(j\omega)|))
    magnitude_db = 20 * np.log10(np.abs(signal))
    plt.semilogx(freq, magnitude_db, label='Magnitude (dB)')

    # Phase Plot (Angle of H(j\omega) in degrees)
    phase_rad = np.angle(signal) # Returns angle in radians
    phase_deg = np.degrees(phase_rad) # Convert to degrees for standard Bode plot display
    plt.semilogx(freq, phase_deg, label='Phase (degrees)')

    # Add labels and title
    plt.title('Bode Plot')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB) / Phase (degrees)')
    plt.grid(True)
    plt.legend()
    plt.show()

// ... existing code ...
