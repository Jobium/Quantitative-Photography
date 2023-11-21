# Quantitative-Photography
CellProfiler and Python scripts for quantitative analysis of size/shape/colour of plastic debris

These scripts process photographs of plastic debris, detecting any objects present and measuring their size, shape, and average colour.

Written by Dr Joseph Razzell Hollis on 2023-07-24. Details and assessment of the underlying methodology were published by Razzell Hollis et al. in the journal Methods in Ecology & Evolution in 2023 (DOI: TBC). Please cite the methods paper if you adapt this code for your own analysis.

Any updates to this script will be made available online at www.github.com/Jobium/Quantitative-Photography/

Python code requires Python 3.7 (or higher) and the following packages: os glob numpy pandas matplotlib scipy skimage sklearn mpl_toolkits itertools
CellProfiler scripts require the CellProfiler program, available for free from https://cellprofiler.org 

# Notes:
1) Object detection and measurement are done using CellProfiler, two example scripts (files ending '.cpproj') are provided for 2021 and 2022 photo sets. For different photo sets, certain variables may need to be adjusted.
 - images will need to be listed for upload in Images module
 - regular expression in Metadata module will need to be updated to reflect filename structure
 -  Crop size (in pixels) for first Crop module will need to be adjusted to include all objects for detection and exclude any unwanted objects
 -  Object diameter min/max in IdentifyPrimaryObjects module may need to be adjusted to reflect differences in image size/resolution
 -  Threshold scaling factor, lower/upper bounds for threshold in IdentifyPrimaryObjects may need to be adjusted to reflect differences in image brightness & dynamic range
2) Statistical analysis and visualisation of measured object parameters (from CellProfiler) were done using Python script 'object_analysis_v01.py'
 - update variable 'Data_dir' to directory containing calibrated images for processing
 - update variable 'Figure_dir' to directory for saving figure images
 - update variable 'Output_dir' to directory for saving output data files
 - update variable 'Img_Res' to resolution of processed images (in millimeters/pixel)

# Citations:
If you use this script, please cite Razzell Hollis et al., Methods in Ecology & Evolution (2023).
