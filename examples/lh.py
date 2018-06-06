from __future__ import division

import numpy as np

from skylab.core import lh
from skylab.core.lh import LHComponent as LHC
from skylab.core.lh import LHFunction as LHF
from skylab.core.minimizer import LBFGSMinimizer

from scipy.optimize import fmin_l_bfgs_b as minimize

#def signal_pdf(params, events):
    ## Lookup the signal PDF value for each event.
    #return events['energy'] / np.max(events['energy'])

#def signal_fraction(params, events):
    #return params['nsignal'] / len(events)
    
#def background_fraction(params, events):
    #return 1 - signal_fraction(params, events)

#lhf = lh.sum(lh.log(LHC(signal_fraction) * LHC(signal_pdf) + LHC(background_fraction)))

events = np.array([(1.0, 2), (3.0, 4), (5.0, 6)], dtype=[('energy', float), ('y', int)]) 
#params = {'nsignal': 1, 'gamma': 2.5}
#print lhf.evaluate(params, events)

def f(p,ev):
    print 'ev', ev
    print 'p:', p
    return p['x']**2 + p['c']

lhf = LHF(f)
#lhf = LHC(lambda p,ev: p['x']**2 + 1).as_function()
lhf_prime = LHC(lambda p,ev: 2*p['x']).as_function()

# Define the parameters for the LH functions.
lhf.params.def_param(name='x', initial=1., isconst=False, valmin=-2., valmax=2.)
lhf.params.def_param(name='c', initial=2.6, isconst=True)



print lhf.params
print lhf.params.variable_initials

m = LBFGSMinimizer(lhf, [lhf_prime])


#(xmin, fmin, d) = minimize(m.get_minimizer_lh_func(), lhf.params.initials, fprime=m.get_minimizer_partial_derivatives_func(), args=(events,), bounds=lhf.params.bounds, approx_grad=False)
(xmin, fmin, d) = m.minimize(events)
print xmin
print fmin
print d 
