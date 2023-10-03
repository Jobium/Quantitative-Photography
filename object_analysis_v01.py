"""
====================================================

This script imports measured parameters of plastic debris (size, shape, colour) from CellProfiler output and visualises the data

====================================================
"""

# ==================================================
# import necessary Python packages

import os                       # for creating output folders
import glob                     # for handling file directories
import numpy as np              # for array manipulation
import pandas as pd             # for dataframe manipulation
import matplotlib.pyplot as plt # for plotting results

from scipy import stats         # statistical functions
from skimage import color       # for conversion betweeen colour spaces
from scipy.optimize import curve_fit        # for fitting colour coordinates
from sklearn.decomposition import PCA       # for doing Principal Component Analysis
from mpl_toolkits.mplot3d import Axes3D     # for generating 3D plots
from mpl_toolkits.axes_grid1 import make_axes_locatable     # for adding colorbars to plots
from itertools import combinations          # for iterating over combinations

# ==================================================
# user-defined variables

# directory of folder containing CellProfiler output spreadsheet containing object measurements
Data_dir = './data/*.csv'     # use '*' as a wildcard, or '**' for multiple layers of subfolders

# directory for folder where figures will be saved
Figure_dir = './figures/'

# directory for folder where processed data will be saved
Output_dir = './output/'

# spatial resolution of image (in um/pixel)
Img_Res = 0.071

Shape_analysis = True   # plot size, shape parameters
Colour_analysis = True  # plot colour coordinates
Corr_plot = True        # plot parameter correlation
Log_plots = True        # plot correlation in log space

Statistics = True           # do statistical tests on datasets by year
Apply_Bonferroni = False    # apply Bonferroni correction to p-values

# list of object IDs to ignore (false positives)
FPs = []

# colours for plotting
Color_list =  ["#ddcc77", "#c41149", "#332288", "#e78500", "#88ccee", "#ff00ca"]

"""
# ==================================================
# define functions for fitting colour coordinates
# ==================================================
"""

def f_polynomial(x, *params):
    # function for generating a polynomial curve
    y = params[0]
    for i in range(1, len(params)):
        y += params[i] * x**i
    return y

def fit_polynomial(x, y, max_order=5, debug=False):
    # function for fitting 2D data points with a polynomial curve
    if len(x) > int(max_order):
        guess = np.zeros((int(max_order)))
    else:
        guess = np.zeros_like(x)
    if debug == True:
        print("        initial parameters: ", guess)
    # run the curve fitting algorithm:
    fit_coeffs, fit_covar = curve_fit(f_polynomial, x, y, p0=guess)
    if debug == True:
        print("        fitted parameters:", fit_coeffs)
    return fit_coeffs, fit_covar

"""
# ==================================================
# set up data storage dicts
# ==================================================
"""

new_cols = ['Bird', 'Date', 'Img ID', 'Area', 'Convex Area', 'Max Feret', 'Min Feret', 'Major Axis', 'Minor Axis','Compactness', 'Solidity',  'Eccentricity', 'Red', 'Green', 'Blue']
cellprofiler_cols = ['Metadata_Bird', 'Metadata_Date', 'Metadata_PhotoID', 'AreaShape_Area', 'AreaShape_ConvexArea', 'AreaShape_MaxFeretDiameter', 'AreaShape_MinFeretDiameter', 'AreaShape_MajorAxisLength', 'AreaShape_MinorAxisLength', 'AreaShape_Compactness', 'AreaShape_Solidity', 'AreaShape_Eccentricity', 'Intensity_MeanIntensity_CropRed', 'Intensity_MeanIntensity_CropGreen', 'Intensity_MeanIntensity_CropBlue']

obj_data = pd.DataFrame(columns=new_cols+['Year'])

print()
print("parameter columns in use:", obj_data.columns.values)

"""
# ==================================================
# import CellProfiler data
# ==================================================
"""

print()
print("finding datasets for import...")

data_dirs = sorted(glob.glob(Data_dir, recursive=True))

print()
print("datasets found:", len(data_dirs))

for data_dir in data_dirs:
    while True:
        try:
            file_name = data_dir.split("/")[-1]
            subfolders = "/".join(data_dir.split("/")[2:-1])
            print()
            print(f"importing {file_name}...")
            print(f"    {data_dir}")
            
            # import spreadsheet as dataframe
            df = pd.read_csv(data_dir)
            print(f"    dataframe:", np.shape(df))
            print(f"        columns:", len(df.columns))
            print(f"           rows:", len(df.index.values))
            
            # generate object ID numbers for indexing and future reference
            mapper = {}
            for i, index in enumerate(df.index.values):
                mapper[index] = "%s-%s%s" % (df['Metadata_Bird'].iloc[i], df['Metadata_Bag'].iloc[i], str(df['ObjectNumber'].iloc[i]).zfill(3))
            df = df.rename(index=mapper)
            
            # check all necessary columns are present in dataset
            check = [col in df.columns for col in cellprofiler_cols]
            if np.all(check) == True:
                # proceed with data import
                
                # trim dataframe to relevant columns only and rows with valid metadata
                sort = pd.isna(df['Metadata_Bird'])
                df_clean = df[cellprofiler_cols][~sort]
                
                # only include rows that are not in False Positive list
                sort = np.asarray([i for i in df_clean.index.values if i not in FPs])
                print(sort)
                df_clean = df_clean.loc[sort]
                
                # rename columns to fit standard naming (new_cols)
                mapper = {}
                for cpcol, pycol in zip(cellprofiler_cols, new_cols):
                    mapper[cpcol] = pycol
                df_clean = df_clean.rename(columns=mapper)
                
                # rescale based on pixel resolution
                for key in ['Max Feret', 'Min Feret', 'Major Axis', 'Minor Axis']:
                    df_clean[key] *= Img_Res
                for key in ['Area', 'Convex Area']:
                    df_clean[key] *= Img_Res**2
                    
                # extract year data from bird IDs
                print(int(df_clean['Bird'].iloc[0].split("-")[1]))
                df_clean['Year'] = [int(df_clean['Bird'].iloc[i].split("-")[1]) for i in range(len(df_clean.index.values))]
                
                # add to pre-existing dataframe
                obj_data = pd.concat([obj_data, df_clean])
                print(np.shape(obj_data))
                print("successfully imported!")
                break
            else:
                print("    dataset is missing columns:", np.asarray(cellprofiler_cols)[~np.asarray(check)])
                print(f"cannot proceed with data import for {file_name}!")
                break
        except Exception as e:
            # if anything goes wrong, print exception error and move on to next dataset
            print("something went wrong! Exception:", e)
            break

# create a list of unique years represented in dataset
Years = list(np.unique(obj_data['Year']))

print()
print("Datasets imported:", Years)

"""
# ==================================================
# object size/shape analysis
# ==================================================
"""

if Shape_analysis == True:
    # plot boxplots for object size, by year
    plt.figure(figsize=(8,4))
    for i, key in enumerate(['Major Axis', 'Max Feret', 'Minor Axis', 'Min Feret']):
        plt.subplot(2, 2, i+1)
        plt.title(key)
        temp = []
        labels = []
        for year in Years:
            sort = np.ravel(np.where((obj_data['Year'] == year) & (~pd.isna(obj_data[key]))))
            if len(sort) > 0 and np.isnan(year) == False:
                temp.append(obj_data[key][sort])
                labels.append("%0.0f" % year)
        plt.boxplot(temp, vert=False, widths=0.75, showfliers=False, labels=labels)
        xmin = np.amin(np.concatenate(temp))
        xmax = np.amax(np.concatenate(temp))
        plt.xlim(xmin - 0.05*(xmax-xmin), xmax + 0.05*(xmax-xmin))
    plt.tight_layout()
    plt.savefig("%ssize_boxplots.png" % (Figure_dir), dpi=300)
    plt.show()
    # plot as violinplot
    plt.figure(figsize=(8,4))
    for i, key in zip(range(4), ['Major Axis', 'Max Feret', 'Minor Axis', 'Min Feret']):
        plt.subplot(2, 2, i+1)
        plt.title(key)
        temp = []
        labels = []
        for year in Years:
            sort = np.ravel(np.where((obj_data['Year'] == year) & (~pd.isna(obj_data[key]))))
            if len(sort) > 0 and np.isnan(year) == False:
                temp.append(obj_data[key][sort])
                labels.append("%0.0f" % year)
        plt.violinplot(temp, vert=False, showmedians=True, widths=0.75)
        xmin = np.amin(np.concatenate(temp))
        xmax = np.amax(np.concatenate(temp))
        plt.xlim(xmin - 0.05*(xmax-xmin), xmax + 0.05*(xmax-xmin))
        plt.yticks(range(1,len(labels)+1), labels)
    plt.tight_layout()
    plt.savefig("%ssize_violinplots.png" % (Figure_dir), dpi=300)
    plt.show()

    # plot histograms for object size
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    for i, key in zip(range(4), ['Major Axis', 'Minor Axis']):
        plt.subplot(1, 2, i+1)
        hists = []
        labels = []
        colors = []
        for year in Years:
            sort = np.ravel(np.where((obj_data['Year'] == year) & (~pd.isna(obj_data[key]))))
            if len(sort) > 0 and np.isnan(year) == False:
                hists.append(obj_data[key][sort])
                labels.append("%0.0f" % (year))
                colors.append(Color_list[Years.index(year)])
        plt.hist(hists, range=(0,50), bins=50, color=colors, label=labels)
        plt.xlabel(key)
        plt.legend()
        plt.minorticks_on()
    plt.tight_layout()
    plt.savefig("%sparameter_histograms.png" % (Figure_dir), dpi=300)
    plt.show()
    
    # plot elliptical approximation vs Feret diameters
    plt.figure(figsize=(8,6))
    # plot major axis vs max Feret diameters
    ax1 = plt.subplot(221)
    ax1.set_title("Major Axis vs Max Feret")
    hists = []
    labels = []
    colors = []
    for year in Years:
        sort = np.ravel(np.where((obj_data['Year'] == year) & (~pd.isna(obj_data['Max Feret']))))
        if len(sort) > 0 and np.isnan(year) == False:
            ax1.plot(obj_data['Max Feret'][sort], obj_data['Major Axis'][sort], 'o', c=Color_list[Years.index(year)], label="%0.f" % (year), alpha=0.1)
            hists.append(obj_data['Major Axis'][sort]/obj_data['Max Feret'][sort])
            labels.append("%0.0f" % (year))
            colors.append(Color_list[Years.index(year)])
    ax1.plot([np.amin(obj_data['Max Feret']), np.amax(obj_data['Max Feret'])], [np.amin(obj_data['Max Feret']), np.amax(obj_data['Max Feret'])], 'k:')
    ax1.set_xlabel("Maximum Feret Diameter (mm)")
    ax1.set_ylabel("Elliptical Major Axis (mm)")
    ax1.legend()
    # plot histogram of major/max ratio
    ax1_hists = plt.subplot(222)
    ax1_hists.hist(hists, range=(0.75, 1.25), bins=25, color=colors, label=labels)
    ax1_hists.set_xlabel("Major Axis/Max Feret Ratio")
    ax1_hists.set_ylabel("Count")
    ax1_hists.set_xlim(0.75, 1.25)
    # plot minor axis vs min Feret diameters
    ax2 = plt.subplot(223)
    ax2.set_title("Minor Axis vs Min Feret")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    hists = []
    labels = []
    colors = []
    for year in Years:
        sort = np.ravel(np.where((obj_data['Year'] == year) & (~pd.isna(obj_data['Min Feret']))))
        if len(sort) > 0 and np.isnan(year) == False:
            ax2.plot(obj_data['Min Feret'][sort], obj_data['Minor Axis'][sort], 'o', c=Color_list[Years.index(year)], label="%0.f" % (year), alpha=0.1)
            hists.append(obj_data['Major Axis'][sort]/obj_data['Max Feret'][sort])
            labels.append("%0.0f" % (year))
            colors.append(Color_list[Years.index(year)])
    ax2.plot([np.amin(obj_data['Min Feret']), np.amax(obj_data['Min Feret'])], [np.amin(obj_data['Min Feret']), np.amax(obj_data['Min Feret'])], 'k:')
    ax2.set_xlabel("Minimum Feret Diameter (mm)")
    ax2.set_ylabel("Elliptical Minor Axis (mm)")
    ax2.legend()
    # plot histogram of major/max ratio
    ax2_hists = plt.subplot(224)
    ax2_hists.hist(hists, range=(0.75, 1.25), bins=25, color=colors, label=labels)
    ax2_hists.set_xlabel("Minor Axis/Min Feret Ratio")
    ax2_hists.set_ylabel("Count")
    ax2_hists.set_xlim(0.75, 1.25)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig("%sferet-vs-elliptical.png" % (Figure_dir), dpi=300)
    plt.show()
    
    # plot distribution for each shape parameter
    keys = ['Major Axis', 'Minor Axis', 'Max Feret', 'Min Feret', 'Area', 'Compactness', 'Solidity', 'Eccentricity']
    plt.figure(figsize=(12,12))
    for i, key in enumerate(keys):
        print()
        print(key)
        sort = np.ravel(np.where(~pd.isna(obj_data[key])))
        print("%s, all data: %0.f total values found" % (key, np.size(sort)))
        print("    IQR =    %0.2f" % (stats.iqr(obj_data[key][sort])))
        print("    mean =   %0.2f +/- %0.2f" % (np.mean(obj_data[key][sort]), np.std(obj_data[key][sort])))
        print("    median = %0.2f" % (np.median(obj_data[key][sort])))
        skew = stats.skew(obj_data[key][sort])
        print("    skew =   %0.2f" % (skew))
        plt.subplot(4, 2, i+1)
        xrange = (np.amin(obj_data[key]), np.amax(obj_data[key]))
        hists = []
        labels = []
        colors = []
        for year in Years:
            sort = np.ravel(np.where((obj_data['Year'] == year) & (~pd.isna(obj_data[key]))))
            if len(sort) > 0 and np.isnan(year) == False:
                hists.append(obj_data[key][sort])
                labels.append("%0.0f" % (year))
                colors.append(Color_list[Years.index(year)])
                print("    %s, %s: %0.f values found" % (key, year, np.size(sort)))
                print("        IQR =    %0.2f" % (stats.iqr(obj_data[key][sort])))
                print("        mean =   %0.2f +/- %0.2f" % (np.mean(obj_data[key][sort]), np.std(obj_data[key][sort])))
                print("        median = %0.2f" % (np.median(obj_data[key][sort])))
                skew = stats.skew(obj_data[key][sort])
                print("        skew =   %0.2f" % (skew))
        plt.hist(hists, range=xrange, bins=50, color=colors, label=labels)
        plt.xlim(xrange)
        plt.xlabel(key)
        plt.legend()
        plt.minorticks_on()
        if Statistics == True and len(labels) > 1:
            # do Kolmogorov-Smirnov tests
            test_count = len(list(combinations(labels, 2))) * len(keys)
            print()
            print("Kolmogorov-Smirnov testing of %s values, by year" % key)
            for i1 in range(len(labels)):
                ks_stats, pval = stats.ks_2samp(hists[i1], np.concatenate(hists))
                print("    %s vs all data: p-val = %s" % (labels[i1], pval))
                if Apply_Bonferroni == True:
                    pval *= test_count
                    print("        after Bonferroni correction: p = %s" % (pval))
                if pval < 0.05:
                    print("        >95% confidence that distributions are distinct")
                else:
                    print("        insufficient confidence to reject null hypothesis, distributions may be the same")
                for i2 in range(i1+1, len(labels)):
                    ks_stats, pval = stats.ks_2samp(hists[i1], hists[i2])
                    print("    %s vs %s: p-val = %s" % (labels[i1], labels[i2], pval))
                    if Apply_Bonferroni == True:
                        pval *= test_count
                        print("        after Bonferroni correction: p = %s" % (pval))
                    if pval < 0.05:
                        print("        >95% confidence that distributions are distinct")
                    else:
                        print("        insufficient confidence to reject null hypothesis, distributions may be the same")
    plt.tight_layout()
    plt.savefig("%sparameter_distributions.png" % (Figure_dir), dpi=300)
    plt.show()
    
if Shape_analysis == True and Statistics == True:
    # do log-normality test of length, width parameters
    plt.figure(figsize=(8,8))
    plt.subplot(211)
    plt.title("Log(Major Axis)")
    xrange = (np.log(1), np.ceil(np.log(np.amax(obj_data['Major Axis']))))
    hists = []
    colors = []
    labels = []
    for year in Years:
        sort = np.ravel(np.where((obj_data['Year'] == year) & (~pd.isna(obj_data['Major Axis']))))
        if len(sort) > 0 and np.isnan(year) == False:
            hists.append(np.log(obj_data['Major Axis'][sort]))
            colors.append(Color_list[Years.index(year)])
            labels.append("%0.f" % year)
            k2, p = stats.normaltest(hists[-1])
            print("    %0.f major axis log-normal test:" % year, p)
    plt.hist(hists, range=xrange, bins=50, color=colors, label=labels)
    sort = np.ravel(np.where(~pd.isna(obj_data['Major Axis'])))
    k2, p = stats.normaltest(np.log(obj_data['Major Axis'][sort]))
    print("all major axis data log-normal test:", p)
    plt.subplot(212)
    plt.title("Log(Minor Axis)")
    xrange = (np.log(1), np.ceil(np.log(np.amax(obj_data['Minor Axis']))))
    hists = []
    colors = []
    labels = []
    for year in Years:
        sort = np.ravel(np.where((obj_data['Year'] == year) & (~pd.isna(obj_data['Minor Axis']))))
        if len(sort) > 0 and np.isnan(year) == False:
            hists.append(np.log(obj_data['Minor Axis'][sort]))
            colors.append(Color_list[Years.index(year)])
            labels.append("%0.f" % year)
            k2, p = stats.normaltest(hists[-1])
            print("    %0.f minor axis log-normal test:" % year, p)
    plt.hist(hists, range=xrange, bins=50, color=colors, label=labels)
    sort = np.ravel(np.where(~pd.isna(obj_data['Minor Axis'])))
    k2, p = stats.normaltest(np.log(obj_data['Minor Axis'][sort]))
    print("all minor axis log-normal test:", p)
    plt.tight_layout()
    plt.savefig("%ssize_lognormal-test.png" % (Figure_dir), dpi=300)
    plt.show()
    
    # calculate covariances between parameters
    correlation_matrix = obj_data[keys].corr()
    print()
    print("linear correlation matrix of raw data:")
    print(correlation_matrix)
    
    plt.figure(figsize=(8,4))
    ax1 = plt.subplot(121)
    ax1.set_title("Linear Correlation")
    im = ax1.imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
    ax1.set_xticks(np.arange(len(keys)))
    ax1.set_xticklabels(keys, rotation='vertical')
    ax1.set_yticks(np.arange(len(keys)))
    ax1.set_yticklabels(keys)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    correlation_matrix = np.log(obj_data[keys]).corr()
    print()
    print("linear correlation matrix of log data:")
    print(correlation_matrix)
    
    ax2 = plt.subplot(122)
    ax2.set_title("Log-log Correlation")
    im = ax2.imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
    ax2.set_xticks(np.arange(len(keys)),)
    ax2.set_xticklabels(keys, rotation='vertical')
    ax2.set_yticks(np.arange(len(keys)))
    ax2.set_yticklabels(keys)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig("%sparameter_correlation.png" % (Figure_dir), dpi=300)
    plt.show()

    # plot all parameters against one another
    Corr_plot = True
    Log_plots = True

    if Corr_plot == True:
        pairs = combinations(range(len(keys)), 2)
        plt.figure(figsize=(2.5*(len(keys)-1), 2.5*(len(keys)-1)))
        for xi, yi in pairs:
            ### print(len(params)-1 - xi, len(params)-yi)
            x_key = keys[xi]
            y_key = keys[yi]
            plt.subplot2grid((len(keys), len(keys)), (len(keys)-1-yi, len(keys)-2-xi))
            ### plt.title("%s v %s" % (y_param, x_param))
            if Log_plots == True:
                plt.xlim(np.floor(np.log(np.amin(obj_data[x_key][sort]))), np.ceil(np.log(np.amax(obj_data[x_key][sort]))))
                plt.ylim(np.floor(np.log(np.amin(obj_data[y_key][sort]))), np.ceil(np.log(np.amax(obj_data[y_key][sort]))))
            else:
                plt.xlim(np.amin(obj_data[x_key][sort]), np.amax(obj_data[x_key][sort]))
                plt.ylim(np.amin(obj_data[y_key][sort]), np.amax(obj_data[y_key][sort]))
            if yi == xi+1:
                if Log_plots == True:
                    plt.xlabel("$ln$(%s)" % x_key)
                    plt.ylabel("$ln$(%s)" % y_key)
                else:
                    plt.xlabel(x_key)
                    plt.ylabel(y_key)
            plt.xticks([])
            plt.yticks([])
            hists = []
            labels = []
            colours = []
            for year in Years:
                sort = np.ravel(np.where((obj_data['Year'] == year) & (~pd.isna(obj_data[x_key])) & (~pd.isna(obj_data[y_key]))))
                x_temp = obj_data[x_key][sort]
                y_temp = obj_data[y_key][sort]
                if Log_plots == True:
                    x_temp = np.log(x_temp)
                    y_temp = np.log(y_temp)
                if xi == yi:
                    hists.append(x_temp)
                    labels.append("%0.0f" % (year))
                    colours.append(Color_list[Years.index(year)])
                else:
                    plt.plot(x_temp, y_temp, 'o', c=Color_list[Years.index(year)], alpha=0.1, label="%0.0f" % (year))
            if yi == xi:
                xlim = plt.xlim()
                hist_vals, hist_bins, patches = plt.hist(hists, range=xlim, bins=40)
                plt.ylim(0, 1.1*np.amax(hist_vals))
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        text = ""
        if Log_plots == True:
            text = "_log"
        plt.savefig("%scorr_plots%s.png" % (Figure_dir, text), dpi=300)
        plt.show()
        
"""
# ==================================================
# colour analysis
# ==================================================
"""

if Colour_analysis == True:
    print()
    print("doing colour analysis")
    
    # get RGB data for reference colours
    RGB_ref_labels = ['White', 'Black', 'Red', 'Yellow', 'Green', 'Cyan', 'Blue', 'Magenta']
    RGB_refs = np.asarray([[0., 0., 0.], [1., 1., 1.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [0., 1., 1.], [0., 0., 1.], [1., 0., 1.]])
    
    # get RGB data for objects
    RGBs = np.vstack([obj_data['Red'], obj_data['Green'], obj_data['Blue']])
    RGBs = RGBs.transpose()
    print(np.shape(RGBs))
    
    # convert to CIELAB coordinates
    LAB_refs = color.rgb2lab(RGB_refs, illuminant='D50')
    LABs = color.rgb2lab(RGBs, illuminant='D50')
    
    # 3D plot of RGB values
    fig = plt.figure(figsize=(8,8))
    ax1 = Axes3D(fig)
    ax1.set_title("sRGB Colour Space")
    ax1.set_xlabel("R")
    ax1.set_ylabel("G")
    ax1.set_zlabel("B")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.scatter(RGB_refs[:,0], RGB_refs[:,1], RGB_refs[:,2], c='w', edgecolors=RGB_refs)
    ax1.scatter(RGBs[:,0], RGBs[:,1], RGBs[:,2], c=RGBs)
    fig.savefig("%sRGB_3D.png" % (Figure_dir), dpi=300)
    fig.show()
    
    # 3D plot of LAB values
    fig = plt.figure(figsize=(8,8))
    ax1 = Axes3D(fig)
    ax1.set_title("CIELAB Colour Space")
    ax1.set_xlabel("a*")
    ax1.set_ylabel("b*")
    ax1.set_zlabel("L*")
    ax1.scatter(LAB_refs[:,1], LAB_refs[:,2], LAB_refs[:,0], c='w', edgecolors=RGB_refs)
    ax1.scatter(LABs[:,1], LABs[:,2], LABs[:,0], c=RGBs)
    fig.savefig("%sCIELAB_3D.png" % (Figure_dir), dpi=300)
    fig.show()
    
    for year in Years:
        # 3D plots by year
        sort = np.ravel(np.where(obj_data['Year'] == year))
        
        # 3D plot of RGB values
        fig = plt.figure(figsize=(8,8))
        ax1 = Axes3D(fig)
        ax1.set_title("sRGB Colour Space")
        ax1.set_xlabel("R")
        ax1.set_ylabel("G")
        ax1.set_zlabel("B")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_zlim(0, 1)
        ax1.scatter(RGB_refs[:,0], RGB_refs[:,1], RGB_refs[:,2], c='w', edgecolors=RGB_refs)
        ax1.scatter(RGBs[sort,0], RGBs[sort,1], RGBs[sort,2], c=RGBs[sort])
        fig.savefig("%sRGB_3D_%0.f.png" % (Figure_dir, year), dpi=300)
        fig.show()

        # 3D plot of LAB values
        fig = plt.figure(figsize=(8,8))
        ax1 = Axes3D(fig)
        ax1.set_title("CIELAB Colour Space")
        ax1.set_xlabel("a*")
        ax1.set_ylabel("b*")
        ax1.set_zlabel("L*")
        ax1.scatter(LAB_refs[:,1], LAB_refs[:,2], LAB_refs[:,0], c='w', edgecolors=RGB_refs)
        ax1.scatter(LABs[sort,1], LABs[sort,2], LABs[sort,0], c=RGBs[sort])
        fig.savefig("%sCIELAB_3D_%0.f.png" % (Figure_dir, year), dpi=300)
        fig.show()
        
    
    # ==================================================
    # Principal Component Analysis
    
    print()
    print("running PCA on colour data")
    
    # prepare dataframe for PCA
    temp = pd.DataFrame(RGBs, columns=['R', 'G', 'B'], index=obj_data.index.values)
    check = np.where((np.isnan(temp['R']) == False) & (np.isnan(temp['G']) == False) & (np.isnan(temp['B']) == False))
    temp = temp.iloc[check]
    print("nan check:", np.any(np.isnan(temp)))

    # run PCA on RGB colour coordinates
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(temp)
    principal_frame = pd.DataFrame(data=principalComponents, columns=['principal component '+str(i+1) for i in range(pca.n_components_)])
    print("features:", pca.n_features_)
    print("components:", pca.n_components_)
    print("samples:", pca.n_samples_)
    print('Explained variation per principal component:')
    for i in range(pca.n_components_):
        print("    component %d: %0.3f" % (i+1, pca.explained_variance_ratio_[i]))
        print("        loading vector:", pca.components_[i])

    # fit PCA coordinates (axes 1 and 2) with a polynomial
    fit_coeffs, fit_covar = fit_polynomial(principal_frame['principal component 1'], principal_frame['principal component 2'], 5)
    x_temp = np.linspace(np.amin(principal_frame['principal component 1']), np.amax(principal_frame['principal component 1']), 1000)
    y_temp = f_polynomial(x_temp, *fit_coeffs)
    
    # plot results for RGB space
    plt.figure(figsize=(12,12))
    ax1 = plt.subplot(221)
    ax1.set_title("PCA of object colour (in sRGB)")
    ax1.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
    ax1.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
    refs = pca.transform(RGB_refs)
    ax1.scatter(refs[:,0], refs[:,1], c='w', edgecolors=RGB_refs)
    ax1.scatter(principal_frame['principal component 1'], principal_frame['principal component 2'], c=RGBs[check])
    ax1.plot(x_temp, y_temp, 'r:')
    ax1.grid()
    x_lim = ax1.get_xlim()
    y_lim = ax1.get_ylim()
    
    # plot 2D histogram for RGB space
    ax3 = plt.subplot(223)
    ax3.set_title("2D Histogram of RGB PCA")
    ax3.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
    ax3.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
    h, xedges, yedges, im = ax3.hist2d(principal_frame['principal component 1'], principal_frame['principal component 2'], bins=(40,40), range=(x_lim, y_lim))
    ax3.plot(x_temp, y_temp, 'r:')
    ax3.grid()
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # repeat analysis using CIELAB colour coordinates
    temp = pd.DataFrame(LABs, columns=['R', 'G', 'B'], index=obj_data.index.values)
    check = np.where((np.isnan(temp['R']) == False) & (np.isnan(temp['G']) == False) & (np.isnan(temp['B']) == False))
    temp = temp.iloc[check]
    print("nan check:", np.any(np.isnan(temp)))
    
    # run PCA on CIELAB colour coordinates
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(temp)
    principal_frame = pd.DataFrame(data=principalComponents, columns=['principal component '+str(i+1) for i in range(pca.n_components_)])
    print("features:", pca.n_features_)
    print("components:", pca.n_components_)
    print("samples:", pca.n_samples_)
    print('Explained variation per principal component:')
    for i in range(pca.n_components_):
        print("    component %d: %0.3f" % (i+1, pca.explained_variance_ratio_[i]))
        print("        loading vector:", pca.components_[i])

    # fit PCA coordinates with polynomial
    fit_coeffs, fit_covar = fit_polynomial(principal_frame['principal component 1'], principal_frame['principal component 2'], 5)
    x_temp = np.linspace(np.amin(principal_frame['principal component 1']), np.amax(principal_frame['principal component 1']), 1000)
    y_temp = f_polynomial(x_temp, *fit_coeffs)
    print(np.shape(x_temp), np.shape(y_temp))
    
    # plot results for CIELAB
    ax2 = plt.subplot(222)
    ax2.set_title("PCA of object colour (in CIELAB)")
    ax2.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
    ax2.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
    refs = pca.transform(LAB_refs)
    ax2.scatter(refs[:,0], refs[:,1], c='w', edgecolors=RGB_refs)
    ax2.scatter(principal_frame['principal component 1'], principal_frame['principal component 2'], c=RGBs[check])
    ax2.plot(x_temp, y_temp, 'r:')
    ax2.grid()
    x_lim = ax2.get_xlim()
    y_lim = ax2.get_ylim()
    
    # plot 2D histogram
    ax4 = plt.subplot(224)
    ax4.set_title("2D Histogram of LAB PCA")
    ax4.set_xlabel('Principal Component 1 (%0.1f %%)' % (100*pca.explained_variance_ratio_[0]))
    ax4.set_ylabel('Principal Component 2 (%0.1f %%)' % (100*pca.explained_variance_ratio_[1]))
    h, xedges, yedges, im = ax4.hist2d(principal_frame['principal component 1'], principal_frame['principal component 2'], bins=(40,40), range=(x_lim, y_lim))
    ax4.plot(x_temp, y_temp, 'r:')
    ax4.grid()
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig("%scolour_PCA.png" % (Figure_dir))
    plt.show()