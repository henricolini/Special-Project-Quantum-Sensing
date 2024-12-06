import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from collections import Counter

class PeakAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sheet_names = pd.ExcelFile(self.file_path).sheet_names
        self.green_laser_power = None
        self.common_peaks_x = None
        self.common_peaks_y = None

    def analyze_sheet(self, sheet_name):
        # Read the data for the specified sheet
        data = pd.read_excel(self.file_path, sheet_name=sheet_name)
        x = data['Time - Plot 0'].values
        y = data['Amplitude - Plot 0'].values
        inverted_y = -y

        # Counter to store the top 3 peaks across distances
        peak_counts = Counter()

        # Loop through distance values from 0 to 10
        for distance in range(1, 11):
            peaks, _ = find_peaks(inverted_y, distance=distance)
            peak_heights = y[peaks]
            top_three_indices = peaks[np.argsort(peak_heights)[:3]]
            peak_counts.update(top_three_indices)

        # Find the 3 most common peaks across distances
        most_common_peaks = [peak for peak, _ in peak_counts.most_common(3)]
        self.common_peaks_x = x[most_common_peaks]
        
        # Calculate true peak heights (max height - y-value at peak)
        max_y = np.max(y)
        self.common_peaks_y = max_y - y[most_common_peaks]

    def plot_common_peaks(self, x, y, sheet_name):
        plt.figure()
        plt.plot(x, y, label='Data')
        plt.plot(self.common_peaks_x, self.common_peaks_y, "x", label='Most Consistent Minima', color='red')
        plt.xlabel('Time - Plot 0')
        plt.ylabel('Amplitude - Plot 0')
        plt.legend()
        plt.title(f'Sheet: {sheet_name}')
        plt.show()

    def get_common_peaks(self):
        return self.common_peaks_x, self.common_peaks_y






    
    
    
  






