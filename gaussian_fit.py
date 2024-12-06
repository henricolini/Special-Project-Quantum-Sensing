import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from scipy import ndimage
from scipy.ndimage import rotate

class LaserImageProfile:
    def __init__(self, barrier_image_path, laser_image_path, pixel_size_um=4.8, power=0.1, real_width=2e-3):
        self.barrier_image_path = barrier_image_path
        self.laser_image_path = laser_image_path
        self.pixel_size_m = pixel_size_um * 1e-6  # Convert pixel size to meters
        self.power = power
        self.real_width = real_width
        self.data_barrier = None
        self.data_laser = None
        self.magnification = None
        self.threshold = 100  # Default threshold for edge detection
        # Analysis results
        self.fwhm = None
        self.radius = None
        self.radius_wo = None
        self.area = None
        self.area_wo = None
        self.intensity = None
        self.intensity_wo = None
        self.r_squared = None
        self.popt = None
        self.x_meters = None
        self.y_meters = None
        self.brightness_horizontal = None
        self.fitted_horizontal = None

    def load_images(self):
        """Load and convert images to grayscale arrays."""
        self.data_barrier = np.array(Image.open(self.barrier_image_path).convert('L'))
        self.data_laser = np.array(Image.open(self.laser_image_path).convert('L'))
        
    def calculate_magnification(self, height=504):
        """Calculate the magnification based on the width of a barrier in the image."""
        if self.data_barrier is None:
            raise ValueError("Barrier image data not loaded. Please call load_images().")
        
        self.data_barrier=rotate(self.data_barrier,3,reshape=False)
        line_profile = self.data_barrier[height, :]
        indices_above_threshold = list(range(561,1074))#np.where(line_profile > self.threshold)[0]

        if len(indices_above_threshold) > 0:
            start_index = indices_above_threshold[0]
            end_index = indices_above_threshold[-1]
            width_in_pixels = end_index - start_index
            width_in_m = width_in_pixels * self.pixel_size_m
            self.magnification = width_in_m / self.real_width

            print(f"Width of the barrier in pixels: {width_in_pixels}")
            print(f"Width of the barrier in meters: {width_in_m:.4f} m")
            print(f"Magnification: {self.magnification:.4f}")
        else:
            print("No barrier detected above the threshold.")
            raise RuntimeError("Barrier detection failed.")

    def analyze_laser_profile_gaussian(self):
        """Analyze the laser profile and return calculated laser parameters."""
        if self.data_laser is None:
            raise ValueError("Laser image data not loaded. Please call load_images().")

        brightest_point = np.unravel_index(np.argmax(self.data_laser, axis=None), self.data_laser.shape)
        y_brightest, x_brightest = brightest_point
        self.brightness_horizontal = self.data_laser[y_brightest, :]

        x_pixels = np.arange(self.data_laser.shape[1])
        self.x_meters = x_pixels * self.pixel_size_m

        initial_guess = [
            np.max(self.brightness_horizontal), 
            x_brightest * self.pixel_size_m, 
            10 * self.pixel_size_m, 
            np.min(self.brightness_horizontal)
        ]

        self.popt, _ = curve_fit(self.gaussian_1d, self.x_meters, self.brightness_horizontal, p0=initial_guess)
        self.fitted_horizontal = self.gaussian_1d(self.x_meters, *self.popt)

        self.fwhm = 2 * np.sqrt(2 * np.log(2)) * self.popt[2]  # FWHM in meters
        self.radius = self.fwhm / 2
        self.radius_wo = 1.699 * self.fwhm / 2
        self.area = np.pi * self.radius ** 2
        self.area_wo = np.pi * self.radius_wo ** 2
        self.intensity = self.power / self.area
        self.intensity_wo = self.power / self.area_wo

        ss_res = np.sum((self.brightness_horizontal - self.fitted_horizontal) ** 2)
        ss_tot = np.sum((self.brightness_horizontal - np.mean(self.brightness_horizontal)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)

        results= {
            "FWHM": self.fwhm,
            "Laser Radius (FWHM)": self.radius,
            "Laser Radius (1/e^2)": self.radius_wo,
            "Laser Intensity (FWHM)": self.intensity,
            "Laser Intensity (1/e^2)": self.intensity_wo,
            "R_squared": self.r_squared,
            "Fit Parameters": self.popt
        }
        print("\nLaser Profile Analysis Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
            

    def analyze_laser_profile_th(self):
        """Analyze the laser profile, fit for varying n, and calculate MSE."""
        if self.data_laser is None:
            raise ValueError("Laser image data not loaded. Please call load_images().")
        
        # Rotate the image by 45 degrees
        self.data_laser = rotate(self.data_laser, 45, reshape=False)
        plt.imshow(self.data_laser, cmap="jet")
    
        # Find the brightest point in the laser data
        brightest_point = np.unravel_index(np.argmax(self.data_laser, axis=None), self.data_laser.shape)
        y_brightest, x_brightest = brightest_point
        self.brightness_vertical = self.data_laser[:, 723]
    
        # Define the vertical axis in meters
        y_pixels = np.arange(self.data_laser.shape[0])
        self.y_meters = y_pixels * self.pixel_size_m
    
        # Initial parameter guess for curve fitting
        initial_guess = [
            np.max(self.brightness_vertical),  # Amplitude
            y_brightest * self.pixel_size_m,   # xo (center position in meters)
            10 * self.pixel_size_m,            # sigma (width parameter)
            np.min(self.brightness_vertical)   # offset (baseline intensity)
        ]
    
        # Lists to store n values and corresponding MSEs
        n_values = []
        mse_values = []
    
        # Loop over varying even values of n
        plt.figure(figsize=(10, 5))
        for n in range(2, 12, 2):  # Step by 2 to ensure n is even
            try:
                # Fit the profile with the current n value
                popt, _ = curve_fit(lambda x, amplitude, xo, sigma, offset: 
                                    self.top_hat(x, amplitude, xo, sigma, offset, n),
                                    self.y_meters, self.brightness_vertical, p0=initial_guess)
                
                # Generate the fitted curve with the optimal parameters
                fitted_vertical = self.top_hat(self.y_meters, *popt, n)
    
                # Calculate the MSE for the current n
                mse = np.mean((self.brightness_vertical - fitted_vertical) ** 2)
                n_values.append(n)
                mse_values.append(mse)
    
                # Plot the fitted curve for this value of n
                plt.plot(self.y_meters, fitted_vertical, label=f'n={n}', linestyle='--')
    
            except RuntimeError:
                print(f"Fit did not converge for n = {n}")
    
        # Plot the original brightness profile
        plt.plot(self.y_meters, self.brightness_vertical, label='Brightness (Vertical)', color='blue')
        plt.title('Vertical Brightness Profile for Varying n')
        plt.xlabel('Vertical Axis (Meters)')
        plt.ylabel('Brightness (Intensity)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
        # Plot MSE vs n
        plt.figure(figsize=(10, 5))
        plt.plot(n_values, mse_values, marker='o', color='red')
        plt.title('Mean Square Error (MSE) vs n')
        plt.xlabel('n (Super Gaussian Order)')
        plt.ylabel('Mean Square Error (MSE)')
        plt.grid(True)
        plt.show()
    
        # Find the optimal n with the minimum MSE
        optimal_n = n_values[np.argmin(mse_values)]
        print(f"Optimal n with minimum MSE: {optimal_n}")
    
        # Re-fit the data using the optimal n value
        popt, _ = curve_fit(lambda x, amplitude, xo, sigma, offset: 
                            self.top_hat(x, amplitude, xo, sigma, offset, optimal_n),
                            self.y_meters, self.brightness_vertical, p0=initial_guess)
    
        # Generate the fitted curve using the optimal n
        fitted_vertical = self.top_hat(self.y_meters, *popt, optimal_n)
    
        # Calculate required parameters
        self.fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2]  # FWHM in meters
        self.radius = self.fwhm / 2
        self.radius_wo = 1.699 * self.fwhm / 2
        self.area = np.pi * self.radius ** 2
        self.area_wo = np.pi * self.radius_wo ** 2
        self.intensity = self.power / self.area
        self.intensity_wo = self.power / self.area_wo
    
        # Calculate R-squared for the fit
        ss_res = np.sum((self.brightness_vertical - fitted_vertical) ** 2)
        ss_tot = np.sum((self.brightness_vertical - np.mean(self.brightness_vertical)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)
    
        # Save the results in a dictionary
        results = {
            "FWHM": self.fwhm,
            "Laser Radius (FWHM)": self.radius,
            "Laser Radius (1/e^2)": self.radius_wo,
            "Laser Intensity (FWHM)": self.intensity,
            "Laser Intensity (1/e^2)": self.intensity_wo,
            "R_squared": self.r_squared,
            "Fit Parameters": popt
        }
    
        # Print the results
        print("\nLaser Profile Analysis Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
    
    @staticmethod
    def top_hat(x, amplitude, xo, sigma, offset, n):
        """Define a Super Gaussian function."""
        return offset + amplitude * np.exp(-2 * ((x - xo) / sigma) ** n)


    def plot_laser_profile(self):
        """Plot the horizontal brightness profile and fitted Gaussian."""
        if self.brightness_horizontal is None or self.fitted_horizontal is None:
            raise ValueError("No analysis data. Please call analyze_laser_profile() first.")

        plt.figure(figsize=(10, 5))
        plt.plot(self.x_meters, self.brightness_horizontal, label='Brightness (Horizontal)', color='blue')
        plt.plot(self.x_meters, self.fitted_horizontal, label='Gaussian Fit (Horizontal)', color='red', linestyle='--')
        plt.axvline(x=self.popt[1], color='green', linestyle=':', label='Center (Brightest Point)')
        plt.title('Horizontal Brightness Profile and Gaussian Fit')
        plt.xlabel('Horizontal Axis (Meters)')
        plt.ylabel('Brightness (Intensity)')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Horizontal fit parameters: Amplitude={self.popt[0]:.2f}, Center={self.popt[1]:.2f}, Sigma={self.popt[2]:.2f}, Offset={self.popt[3]:.2f}")
    
    
    @staticmethod
    def gaussian_1d(x, amplitude, xo, sigma, offset):
        """Define a 1D Gaussian function."""
        return offset + amplitude * np.exp(-2*((x - xo) ** 2) / (2 * sigma ** 2))

    def plot_3d_gaussian(self):
        """Plot a 3D Gaussian fit for the entire laser profile."""
        if self.data_laser is None:
            raise ValueError("Laser image data not loaded. Please call load_images().")

        # Grid of x and y values in meters around the center
        x_pixels = np.arange(self.data_laser.shape[1])
        y_pixels = np.arange(self.data_laser.shape[0])
        x_meters = x_pixels * self.pixel_size_m
        y_meters = y_pixels * self.pixel_size_m
        X, Y = np.meshgrid(x_meters, y_meters)

        # Initial guess for 2D Gaussian parameters
        brightest_point = np.unravel_index(np.argmax(self.data_laser, axis=None), self.data_laser.shape)
        initial_guess_2d = [
            np.max(self.data_laser), 
            brightest_point[1] * self.pixel_size_m, 
            brightest_point[0] * self.pixel_size_m,
            10 * self.pixel_size_m, 
            10 * self.pixel_size_m, 
            np.min(self.data_laser)
        ]

        # Fit the Gaussian to the entire 2D image
        popt_2d, _ = curve_fit(lambda xy, amp, xo, yo, sig_x, sig_y, offset: self.gaussian_2d(
            xy[0], xy[1], amp, xo, yo, sig_x, sig_y, offset),
            (X.ravel(), Y.ravel()), self.data_laser.ravel(), p0=initial_guess_2d)

        # Generate the fitted 2D Gaussian data
        fitted_2d = self.gaussian_2d(X, Y, *popt_2d)

        # Create a 3D plot of the fitted Gaussian
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, fitted_2d, cmap='viridis', edgecolor='none')
        ax.set_title('3D Gaussian Fit of Laser Profile')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Brightness (Intensity)')
        plt.show()

    @staticmethod
    def gaussian_2d(x, y, amplitude, xo, yo, sigma_x, sigma_y, offset):
        """Define a 2D Gaussian function."""
        return offset + amplitude * np.exp(
            -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2)))
 