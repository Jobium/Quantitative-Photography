# Quantitative-Photography
CellProfiler and Python scripts for quantitative analysis of size/shape/colour of plastic debris

These scripts process photographs of plastic debris, detecting any objects present and measuring their size, shape, and average colour.

Written by Dr Joseph Razzell Hollis on 2023-07-24. Details and assessment of the underlying methodology were published by Razzell Hollis et al. in the journal XXX on DATE (DOI: XXX). Please cite the methods paper if you adapt this code for your own analysis.

Any updates to this script will be made available online at www.github.com/Jobium/Quantitative-Photography/

Python code requires Python 3.7 (or higher) and the following packages: os glob numpy pandas matplotlib scipy skimage sklearn mpl_toolkits itertools

# Notes:
1) Object detection and measurement are done using CellProfiler, two example scripts (files ending '.cpproj') are provided for 2021 and 2022 photo sets
2) Statistical analysis and visualisation of measured object parameters (from CellProfiler) were done using Python script 'object_analysis_v01.py'

# Citations:
If you use this script, please cite Razzell Hollis et al. (2023), JOURNAL, INFO.
