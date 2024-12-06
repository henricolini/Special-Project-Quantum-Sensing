from Plots_Contrast_Slope_Linewisth import ODMRAnalyzer
from gaussian_fit import LaserImageProfile

#Open Files for Data

#odmr excel
file_path_odmr_data = r"C:\Users\kikos\Desktop\Faculdade\5 ano\1º semestre\Special Project Emma_Henrique\Data ODMR Luca pc\20241128\TopHat_laserpower_sweep\ODMR Top_hat.xlsx" # Replace with your file path
#Profile it will fit to a gaussian and calculate intensity
file_path_profile= r"C:\Users\kikos\Desktop\Faculdade\5 ano\1º semestre\Special Project Emma_Henrique\Data ODMR Luca pc\20241128\Gaussian_laserpower_sweep\gaussian_laser_image.bmp"
#Image it will do the calculations to get the magnification to calculate intensity,  you shouldn't change unless we are working in another setup
file_path_calibration_profile= r"C:\Users\kikos\Desktop\Faculdade\5 ano\1º semestre\Special Project Emma_Henrique\Data ODMR Luca pc\20241128\Gaussian_laserpower_sweep\gaussian_edges_image.bmp"

# Load classes
profile_class = LaserImageProfile(file_path_calibration_profile,file_path_profile) #Calculates intensity and plots profile class
analyzer = ODMRAnalyzer(file_path_odmr_data,amp_search_size_gaus=1e-5) #fits odmr data and does the analysis

########################################################  

#Profile_class pre_calculations
#profile_class.load_images() #loads the images, always necessary
#profile_class.calculate_magnification()  # you can adjust `height=309` based on the line of interest, you shouldn't change unless we are working in another setup

#prints intensity and another info about the gaussian beam
#profile_class.analyze_laser_profile_gaussian()
#profile_class.analyze_laser_profile_th()

#Plot the horizontal laser profile and fitted Gaussian
#profile_class.plot_laser_profile()

#Plot the 3D Gaussian fit
#profile_class.plot_3d_gaussian()

########################################################

#analysis of odmr
analyzer.run_analysis()  # processes data
#analyzer.plot_fit()  # Plot fits for raw data
analyzer.plot_slope_fhwh_contr()  # Plot analysis results for different laser power