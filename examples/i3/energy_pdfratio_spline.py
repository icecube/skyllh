# -*- coding: utf-8 -*-

"""Example script how to create an IceCube energy PDF ratio spline object.
"""
from skylab.physics.flux import PowerLawFlux
from skylab.core.parameters import make_linear_parameter_grid_1d
from skylab.i3.pdfratio import I3EnergySigOverBkgPDFRatioSpline
from skylab.i3.signalpdf import SignalI3EnergyPDF
from skylab.i3.backgroundpdf import DataBackgroundI3EnergyPDF

from skylab.plotting.i3.pdfratio import I3EnergySigOverBkgPDFRatioSplinePlotter

from matplotlib import pyplot as plt

ncpu = 6
season_name = 'IC86, 2015-2017'

# Enable multi-core processing globally.
from skylab.core import multiproc
multiproc.NCPU = ncpu

# Define the GFU dataset collection.
from i3skylab.datasets import GFU_v002p01 as datasample
GFU_dsc = datasample.create_dataset_collection("/home/mwolf/data/ana/analyses/gfu/version-002-p01/")

# Get a dataset from the collection and load and prepare its data.
ds = GFU_dsc.get_dataset(season_name).load_and_prepare_data()

# Get the log_energy and sin_dec binning definition from the dataset.
logE_binning = ds.get_binning_definition('log_energy')
sinDec_binning = ds.get_binning_definition('sin_dec')

# Define the flux model.
fluxmodel = PowerLawFlux(Phi0=1., E0=1e3, gamma=2.)

# Define the fit parameter grid for the gamma fit parameter.
gamma_param_grid = make_linear_parameter_grid_1d(name='gamma', low=1., high=4., delta=0.1)

# Create an IceCube specific energy signal PDF object from the monte-carlo data
# and for a given flux model. It needs to be created for a grid of fit
# parameters. In this case we use a power law flux model, and we want to fit
# the spectral index gamma.
signalpdfset = SignalI3EnergyPDF(ds.data_mc,
    logE_binning, sinDec_binning, fluxmodel, fitparams_grid_set=gamma_param_grid)

# Create an IceCube specific energy background PDF object from experimental
# data.
backgroundpdf = DataBackgroundI3EnergyPDF(ds.data_exp,
    logE_binning, sinDec_binning)

# Create the object for the signal over background energy PDF ratio.
energy_pdf_ratio = I3EnergySigOverBkgPDFRatioSpline(signalpdfset, backgroundpdf)

plotter = I3EnergySigOverBkgPDFRatioSplinePlotter(energy_pdf_ratio)

# Create a matplotlib figure and Axes object.
fig = plt.figure()

plotter.plot(fig.add_subplot(311), fitparams={'gamma':1.9})
plotter.plot(fig.add_subplot(312), fitparams={'gamma':2.9})
plotter.plot(fig.add_subplot(313), fitparams={'gamma':3.9})

plt.show()
