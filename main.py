
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
# path='/path/to/mzmlfiles/'
path='/Users/tk2812/py/msfiles/'

# instantiate MS experiment object (detects all mzml files in specified directory)
dataSet=ms.ExpSet(path, ms.msExp)
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
dataSet.exp[1].plt_chromatogram(ctype=['tic', 'bpc', 'xic'], xic_mz=50, xic_ppm=20)

# plot selection of ms level 1 data (using rt and mz window)
# q_noise: quantile probability of noise intensity (typically 0.95 works well)
selection={'mz_min':30, 'mz_max':1300, 'rt_min': 450, 'rt_max': 500}
dataSet.exp[0].vis_spectrum(q_noise=0.95, selection=selection)

# peak picking has three parameters
# nn distance, min_samples, st_adj
dataSet.exp[0].peak_picking(st_adj=6e2, q_noise=0.95, dbs_par={'eps': 0.02, 'min_samples': 2}, selection=selection, plot=False)
fig, ax = dataSet.exp[0].vis_features(selection=selection, rt_bound=0.51, mz_bound=0.1)


# filter feature qc,
# link isotopes, adducts
# input SC and Annie