
import msmate as ms


# python variable types, examples
name = 'test' # string
ppm = 10 # numeric (integer)
ppmf = 10. # numeric (float)
ages = [10, 15, 200] # list of integers
d = {'age': 20, 'sex': 'M'} # dictionary (=lookup table)

# conversion of proprietary format to mzml
# msconvert data/* -o my_output_dir/


## directory of mzml files
path='/path/to/mzmlfiles/'
#path='/Users/tk2812/py/msfiles/'
#path='/Volumes/Backup Plus/Cambridge_RP_POS'

# instantiate MS experiment object (detects all mzml files in specified directory)
dataSet=ms.ExpSet(path, msExp=ms.msExp, ftype='mzML', nmax=4)
# check out the detailed file list using PyCharm variable panel

# read in data
# option 1: read-in only selected experiments
dataSet.read(pattern='DDA')
# option 2: read-in all experiments
#dataSet.read()

# methods for individual spectrum
# plot chromatograms of individual experiments
# indexing starts at zero (not at 1 like in R)
dataSet.files
dataSet.exp[0].plt_chromatogram(ctype=['tic', 'bpc'])
dataSet.exp[1].plt_chromatogram(ctype=['tic', 'bpc', 'xic'], xic_mz=500, xic_ppm=20)

# plot selection of ms level 1 data (using rt and mz window)
# q_noise: quantile probability of noise intensity (typically 0.95 works well)
selection={'mz_min':180, 'mz_max':230, 'rt_min': 360, 'rt_max': 400}
dataSet.exp[0].vis_spectrum(q_noise=0.95, selection=selection)

# peak picking using a density based clustering algorithm
# in this tutorial, the terms cluster and LC-MS feature are used interchangeably
# clustering algorithm has three parameters: eps: minimum data point distance, min_samples, st_adj
# parameter description:
# eps: minimum distance from one point to another. If two points are within eps, then both points areassigned to the same cluster
# st_adj: adjustment factor for the scantime (st) dimension. Higher st_adj values reduce the distance between two data points in time dimension, \
# scatime adjustment is needed to account for different spectrometer parameters (e.g. 10 Hz -> 10 ms level 1 scans per second vs 20 Hz -> 20 ms level 1 scans per second)
# min_samples: minimum of core data points in neighbourhood eps. A core point is a data point that has a minimum of min_samples of core neighbours in its eps neighbour, \
# the min_samples parameter is directly related to the eps parameter

# selection={'mz_min':300, 'mz_max':400, 'rt_min': 450, 'rt_max': 600}

# clustering set A:
dataSet.exp[0].peak_picking(st_adj=6e2, q_noise=0.55, dbs_par={'eps': 0.0031, 'min_samples': 3}, selection=selection, plot=False)
fig, ax = dataSet.exp[0].vis_features(selection=selection, rt_bound=0.51, mz_bound=0.1)
# Number of L1 clusters: 2527
# Number of L2 clusters: 522


# feature summary
dataSet.exp[0].fs.columns


# qc index




# clustering set B:
dataSet.exp[0].peak_picking(st_adj=6e1, q_noise=0.95, dbs_par={'eps': 0.02, 'min_samples': 2}, selection=selection, plot=False)
fig, ax = dataSet.exp[0].vis_features(selection=selection, rt_bound=0.51, mz_bound=0.1)


# clustering set C:
dataSet.exp[0].peak_picking(st_adj=2e2, q_noise=0.85, dbs_par={'eps': 0.02, 'min_samples': 2}, selection=selection, plot=False)
fig, ax = dataSet.exp[0].vis_features(selection=selection, rt_bound=0.51, mz_bound=0.051)

# check performance of peak picking parameters with new sample
dataSet.exp[1].peak_picking(st_adj=2e2, q_noise=0.85, dbs_par={'eps': 0.02, 'min_samples': 2}, selection=selection, plot=False)
fig, ax = dataSet.exp[1].vis_features(selection=selection, rt_bound=0.51, mz_bound=0.051)

# case 1:
# case 1: data points assigned to a feature show high deviations in mz dimension (usually assessed in ppm), which clustering parameter(s) would you need to change to reduce variation in mz dimension?
# case 2: all of the detected features have an elution time > 10 seconds. Which parameter would you need to change to capture features with lower elution times?
# case 3: detected features capture all true signals accurately, but also capture low intensity points that are unlikely to be real signals. What paramter do you need to change?
# case 4: you don't get any clusters, which parameterisation strategy would you run?

# filter feature qc,
# link isotopes, adducts
# input SC and Annie