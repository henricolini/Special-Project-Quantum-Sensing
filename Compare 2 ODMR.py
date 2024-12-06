import pandas as pd
import matplotlib.pyplot as plt

# Load the max slope data (2 files)
slope_top_hat = pd.read_csv('Saves\max_slope_top_hat.csv')
slope_gaussian = pd.read_csv('Saves\max_slope_gaussian1.csv')

# Load the contrast data (2 files)
contrast_top_hat = pd.read_csv('Saves\contrast_top_hat.csv')
contrast_gaussian = pd.read_csv('Saves\contrast_gaussian1.csv')

# Load the linewidth data (2 files)
linewidth_top_hat = pd.read_csv('Saves\linewidth_top_hat.csv')
linewidth_gaussian = pd.read_csv('Saves\linewidth_gaussian1.csv')

# Plotting max slope comparison
plt.figure(figsize=(10, 6))
plt.plot(slope_top_hat['Green Power (mW)'], slope_top_hat['Max Slope (V/Hz)'], label='g1:Intensity=1462kWm', marker='o')
plt.plot(slope_gaussian['Green Power (mW)'], slope_gaussian['Max Slope (V/Hz)'], label='g2:Intensity=1401kWM', marker='o')
plt.xlabel('Green Power (mW)')
plt.ylabel('Max Slope (V/Hz)')
plt.title('Max Slope Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Plotting contrast comparison
plt.figure(figsize=(10, 6))
plt.plot(contrast_top_hat['Green Power (mW)'], contrast_top_hat['Max Slope (V/Hz)'], label='g1:Intensity=1462kWm', marker='o')
plt.plot(contrast_gaussian['Green Power (mW)'], contrast_gaussian['Max Slope (V/Hz)'], label='g2:Intensity=1401kWM', marker='o')
plt.xlabel('Green Power (mW)')
plt.ylabel('Max Contrast (%)')
plt.title('Contrast Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Plotting linewidth comparison
plt.figure(figsize=(10, 6))
plt.plot(linewidth_top_hat['Green Power (mW)'], linewidth_top_hat['Max Slope (V/Hz)'], label='g1:Intensity=1462kWm', marker='o')
plt.plot(linewidth_gaussian['Green Power (mW)'], linewidth_gaussian['Max Slope (V/Hz)'], label='g2:Intensity=1401kWM', marker='o')
plt.xlabel('Green Power (mW)')
plt.ylabel('Linewidth (Hz)')
plt.title('Linewidth Comparison')
plt.legend()
plt.grid(True)
plt.show()

