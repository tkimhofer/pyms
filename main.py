import msmate as ms

# conversion of proprietary format to mzml
# msconvert data/* -o my_output_dir/


## directory of mzml files
path='/path/to/mzmlfiles/'
# path='/Volumes/Backup Plus/Cambridge_RP_POS'

##############
# detect mzml files
##############

# instantiate MS experiment object (detects all mzml files in specified directory)
dataSet=ms.ExpSet(path, msExp=ms.msExp, ftype='mzML', nmax=3)

# you can use a prepared dataSet object for this tutorial (20 RPNEG spectra)
# import pickle
# pickle.dump([dataSet.tic, dataSet.tic_st], open( path+"/first20_tics_msmate.p", "wb" ) )
# dataSet = pickle.load( open("/Volumes/Backup Plus/Cambridge_RP_POS/first20_msmate.p", "rb" ) )

# check out the detailed file list using PyCharm variable panel
# dataSet.files


##############
# read in lc-ms data
##############

# option 1: read-in only selected experiments using pattern matching:
# dataSet.read(pattern='DDA')

# option 2: read-in all experiments
dataSet.read()

##############
# visualise chromatograms ...
##############

# ...for a single experiment (in Python index starts at zero (not at 1 like in R))....
dataSet.exp[0].plt_chromatogram(ctype=['tic', 'bpc'])
dataSet.exp[1].plt_chromatogram(ctype=['tic', 'bpc'])
dataSet.exp[19].plt_chromatogram(ctype=['tic', 'bpc'])

# ... for all experiments - check for significant retention time shifts
dataSet.get_bpc(plot=True)
dataSet.get_tic(plot=True)


##############
# compare chromatogram signals and select representative LC-MS experiment
##############
# peak picking parametmrs should be optimised for a spectrum that is representative for most spectra in a study
# select TWO representative experiments (one for parameter testing, the other one for validation)
# clustering analysis of a chromatogram can be used to find out which spectrum is representative
# in this examples I cluster the base peak chromatograms, since this is less sensitive to noise
dataSet.chrom_dist(ctype='bpc', minle=3)

# visualise distance tree for individual chromatograms
ax = dataSet.chrom_btree('bpc', index=0)
ax1 = dataSet.chrom_btree('bpc', index=1, ax=ax, colour='red')
ax2 = dataSet.chrom_btree('bpc', index=2, ax=ax1, colour='blue')



##############
# perform peak picking with representative sample
##############

# peak picking parameter optimisation
# select ppm window where there are clear signals - can be identified with TIC
dataSet.exp[0].plt_chromatogram(ctype=['tic'])

# plot selection of ms level 1 data (using rt and mz window)
# q_noise: quantile probability of noise intensity (typically 0.95 works well)
selection={'mz_min':60, 'mz_max':400, 'rt_min':300, 'rt_max': 400}
dataSet.exp[0].vis_spectrum(q_noise=0.89, selection={'mz_min':120, 'mz_max':140, 'rt_min':355, 'rt_max': 395})


# peak picking using a density based clustering algorithm
# peak picking parameter description:
# clustering algorithm has four parameters: eps: minimum data point distance, min_samples, st_adj
# eps: minimum distance from one point to another. If two points are within eps, then both points areassigned to the same cluster
# st_adj: adjustment factor for the scantime (st) dimension. Higher st_adj values reduce the distance between two data points in time dimension, \
# scatime adjustment is needed to account for different spectrometer parameters (e.g. 10 Hz -> 10 ms level 1 scans per second vs 20 Hz -> 20 ms level 1 scans per second)
# min_samples: minimum of core data points in neighbourhood eps. A core point is a data point that has a minimum of min_samples of core neighbours in its eps neighbour, \


# clustering paramaters - experiment 1
dataSet.exp[0].peak_picking(st_adj=6e2, q_noise=0.75, dbs_par={'eps': 0.0041, 'min_samples': 5}, selection=selection,
                            qc_par = {'st_len_min': 7, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 30, 'ppm': 15})
dataSet.exp[0].vis_feature_pp(selection=selection,  lev =3)

el=ms.element_list()
calc_mzIsotopes(formula='C3O3H6', el=el)
# for a given molecular species, predict mz & isotopic pattern and search in peak list



self=dataSet.exp[1]

self.vis

# clustering paramaters - experiment 1, different location
selection={'mz_min':100, 'mz_max':800, 'rt_min': 120, 'rt_max': 6000}
selection={'mz_min':0, 'mz_max':1200, 'rt_min': 0, 'rt_max': 6000}
dataSet.exp[0].vis_spectrum(q_noise=0.75, selection=selection)

st_adj=6e2
q_noise=0.75
dbs_par={'eps': 0.0171, 'min_samples': 2}
self=dataSet.exp[0]

dataSet.exp[0].peak(st_adj=6e2,
                            q_noise=0.75,
                            dbs_par={'eps': 0.0171, 'min_samples': 2},
                            selection=selection,
                            qc_par={'st_len_min': 7, 'raggedness': 0.15, 'non_neg': 0.8, 'sId_gap': 0.15, 'sino': 50, 'ppm': 15})


# clustering paramaters - experiment 2
dataSet.exp[0].peak_picking(st_adj=6e2, q_noise=0.75, dbs_par={'eps': 0.0171, 'min_samples': 2}, selection=selection, plot=True)
self=dataSet.exp[0]
dataSet.exp[0].vis_spectrum(q_noise=0.85, selection=selection)
dataSet.exp[1].vis_feature_pp(selection=selection, rt_bound=1, mz_bound=0.1)


##############
# peak pick all of the samples
##############
dataSet.pick_peaks(st_adj=601, q_noise=0.5, dbs_par={'eps': 0.00311, 'min_samples': 2}, selection=selection,  multicore=False)

selection={'mz_min':250, 'mz_max':800, 'rt_min': 10, 'rt_max': 800}
selection={'mz_min':250, 'mz_max':800, 'rt_min': 80, 'rt_max': 120}
selection={'mz_min':350, 'mz_max':550, 'rt_min': 350, 'rt_max': 480}

dataSet.exp[0].vis_feature_pp(selection=selection, rt_bound=0, mz_bound=0)
self=dataSet.exp[0]
dataSet.exp[1].vis_feature_pp(selection=selection, rt_bound=1, mz_bound=0.1)





##############
# vis peaks across spectra
##############

dataSet.vis_peakGroups()
# definition of sample groups and variation limnits to cluster features across samples



##########
# peak grouping /filtering -> what setup is useful to ST/Annie?
# PCA/O-PLS modelling
# mslevel2 and db annot for met id-ing
##########