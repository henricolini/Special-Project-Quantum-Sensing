import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from class_find_peaks import PeakAnalyzer

class ODMRAnalyzer:
    def __init__(self, file_path, peak_search_size_gaussian=2, amp_search_size_gaus=1e-5, std_peaks=1):
        self.file_path = file_path
        self.peak_search_size_gaussian = peak_search_size_gaussian
        self.amp_search_size_gaus = amp_search_size_gaus
        self.std_peaks = std_peaks
        self.sheet_names = pd.ExcelFile(file_path).sheet_names
        self.summary_data = pd.read_excel(file_path, sheet_name=self.sheet_names[0])
        self.green_laser_power = self.summary_data["Green Power (mW)"].values
        self.slope_max = []
        self.fwhm_list = []
        self.contrast_list = []
        self.common_peaks_x = []
        self.common_peaks_y = []
        self.y_fit_all = []  # Store fitted values for optional plotting later
        self.analyzer = PeakAnalyzer(file_path)
        self.current_index = 0  # To track sheet for laser power plotting

    @staticmethod
    def inverted_lorentzian(x, a0, x0, gamma0, a1, x1, gamma1, a2, x2, gamma2, b):
        return -a0 / (1 + ((x - x0) / gamma0) ** 2) - \
               a1 / (1 + ((x - x1) / gamma1) ** 2) - \
               a2 / (1 + ((x - x2) / gamma2) ** 2) + b
    

    def analyze_sheet(self, sheet_name):
        # Analyze and retrieve common peak data for fitting
        self.analyzer.analyze_sheet(sheet_name)
        common_peaks_x, common_peaks_y = self.analyzer.get_common_peaks()

        # Load data
        data = pd.read_excel(self.file_path, sheet_name=sheet_name)
        x = data['Time - Plot 0'].values
        y = data['Amplitude - Plot 0'].values


        # Fit data using initial guesses based on peak analysis
        initial_guess = [
            common_peaks_y[0], common_peaks_x[0], self.std_peaks,
            common_peaks_y[1], common_peaks_x[1], self.std_peaks,
            common_peaks_y[2], common_peaks_x[2], self.std_peaks,
            min(y)
        ]
        bounds_lower = [
            common_peaks_y[0] - self.amp_search_size_gaus, common_peaks_x[0] - self.peak_search_size_gaussian, -np.inf,
            common_peaks_y[1] - self.amp_search_size_gaus, common_peaks_x[1] - self.peak_search_size_gaussian, -np.inf,
            common_peaks_y[2] - self.amp_search_size_gaus, common_peaks_x[2] - self.peak_search_size_gaussian, -np.inf,
            -np.inf
        ]
        bounds_upper = [
            common_peaks_y[0] + self.amp_search_size_gaus, common_peaks_x[0] + self.peak_search_size_gaussian, np.inf,
            common_peaks_y[1] + self.amp_search_size_gaus, common_peaks_x[1] + self.peak_search_size_gaussian, np.inf,
            common_peaks_y[2] + self.amp_search_size_gaus, common_peaks_x[2] + self.peak_search_size_gaussian, np.inf,
            np.inf
        ]
        popt, _ = curve_fit(self.inverted_lorentzian, x, y, p0=initial_guess, maxfev=100000, bounds=(bounds_lower, bounds_upper))
        y_fit = self.inverted_lorentzian(x, *popt)

        # Calculate and store slope max, FWHM, and contrast for current sheet
        self.calculate_metrics(x, y, y_fit, popt)

        # Save the x, y, and y_fit for optional plotting
        self.common_peaks_x.append(common_peaks_x)
        self.common_peaks_y.append(common_peaks_y)
        self.y_fit_all.append((x, y, y_fit))

        self.current_index += 1

    def calculate_metrics(self, x, y, y_fit, popt):
        # Max slope calculation
        slope = np.gradient(y_fit, x)
        self.slope_max.append(max(slope))


        # FWHM calculation
        sigma = popt[2]+popt[5]+popt[8]  # Assuming the sum of the 3 sigmas is representative
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        self.fwhm_list.append(fwhm)

        # Contrast calculation
        contrast = y_fit / max(y_fit)  # Scaled contrast calculation
        contrast_max= (1-min(contrast)) * 100
        self.contrast_list.append(contrast_max)

    def plot_fit(self):
        # Loop over all sheets to plot the fit for each one
        for i, (x, y, y_fit) in enumerate(self.y_fit_all):
            sheet_name = self.sheet_names[i + 1]  # Get sheet name for title, offset by 1 for summary sheet

            plt.figure(figsize=(10, 6))
            plt.plot(x, y - np.mean(y[:10]), marker='o', linestyle='-', label=f'Data ({self.green_laser_power[i]} mW)')
            plt.plot(x, y_fit - np.mean(y_fit[:10]), marker='o', linestyle='-', label='Fitted Gaussian')
            plt.xlabel('Time - Plot 0')
            plt.ylabel('Amplitude - Plot 0')
            plt.legend()
            plt.title(f'Sheet: {sheet_name} - Fitted Gaussian')
            plt.grid(True)
            plt.show()

    def plot_slope_fhwh_contr(self):
        # Plot maximum slope versus laser power and save
         # Save the slope_max and green_laser_power to a CSV file
        data = {
            "Green Power (mW)": self.green_laser_power,  # Assuming it's a list or array of values
            "Max Slope (V/Hz)": self.slope_max
            }

        # Create a DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv("max_slope_gaussian2.csv", index=False)
        
        data = {
            "Green Power (mW)": self.green_laser_power,  # Assuming it's a list or array of values
            "Max Slope (V/Hz)": self.fwhm_list
            }

        # Create a DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv("linewidth_gaussian2.csv", index=False)
        
        data = {
            "Green Power (mW)": self.green_laser_power,  # Assuming it's a list or array of values
            "Max Slope (V/Hz)": self.contrast_list
            }

        # Create a DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv("contrast_gaussian2.csv", index=False)
        
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.green_laser_power, self.slope_max, marker='o', linestyle='-', color='b', label="Max Slope (V/Hz)")
        plt.xlabel('Green Laser Power (mW)')
        plt.ylabel('Maximum Slope (V/Hz)')
        plt.title('Max Slope for Different Laser Powers')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Dual-axis plot for FWHM and Contrast
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(self.green_laser_power, self.fwhm_list, marker='o', linestyle='-', color='b', label='FWHM (Hz)')
        ax1.set_xlabel('Green Laser Power (mW)')
        ax1.set_ylabel('FWHM (Hz)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(self.green_laser_power, self.contrast_list, marker='s', linestyle='-', color='r', label='Max Contrast (%)')
        ax2.set_ylabel('Max Contrast (%)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title('FWHM and Max Contrast for Different Laser Powers')
        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()

    def run_analysis(self):
        for sheet in self.sheet_names[1:]:  # Skips summary sheet
            self.analyze_sheet(sheet)





