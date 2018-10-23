# -*- coding: utf-8 -*-

from skylab.core.py import issequenceof
from skylab.core.detsigeff import DetSigEffImplMethod
from skylab.core.signal_generation import SignalGenerationMethod
from skylab.physics.source import SourceModel
from skylab.physics.flux import FluxModel


class SourceHypoGroup(object):
    """The source hypothesis group class provides a data container to describe
    a group of sources that share the same flux model and detector signal
    efficiency implementation method.
    """
    def __init__(self, sources, fluxmodel, detsigeff_implmethods, sig_gen_method):
        """Constructs a new source hypothesis group.

        Parameters
        ----------
        sources : SourceModel | sequence of SourceModel
            The source or sequence of sources that define the source group.
        fluxmodel : instance of FluxModel
            The FluxModel instance that applies to the list of sources of the
            group.
        detsigeff_implmethods : sequence of DetSigEffImplMethod instances
            The sequence of detector signal efficiency implementation method
            instances, which should be used to create the detector signal
            efficiency for the sources of the group. Each element is the
            detector signal efficiency implementation method for the particular
            dataset, if several datasets are used. If this list contains only
            one implementation method, it should be used for all datasets.
        sig_gen_method : SignalGenerationMethod instance | None
            The instance of SignalGenerationMethod that implements the signal
            generation for the specific detector and source hypothesis. It can
            be set to None, which means, no signal can be generated. Useful for
            data unblinding and data trial generation, where no signal is
            required.
        """
        self.source_list = sources
        self.fluxmodel = fluxmodel
        self.detsigeff_implmethod_list = detsigeff_implmethods
        self.sig_gen_method = sig_gen_method

    @property
    def source_list(self):
        """The list of SourceModel instances for which the group is defined.
        """
        return self._source_list
    @source_list.setter
    def source_list(self, sources):
        if(isinstance(sources, SourceModel)):
            sources = [ sources ]
        if(not issequenceof(sources, SourceModel)):
            raise TypeError('The source_list property must be an instance of SourceModel or a sequence of SourceModel instances!')
        self._source_list = list(sources)

    @property
    def fluxmodel(self):
        """The FluxModel instance that applies to the list of sources of this
        source group.
        """
        return self._fluxmodel
    @fluxmodel.setter
    def fluxmodel(self, fluxmodel):
        if(not isinstance(fluxmodel, FluxModel)):
            raise TypeError('The fluxmodel property must be an instance of '
                'FluxModel!')
        self._fluxmodel = fluxmodel

    @property
    def detsigeff_implmethod_list(self):
        """The list of DetSigEffImplMethod instances, which should be used to
        create the detector signal efficiency for this group of sources. Each
        element is the detector signal efficiency implementation method for
        the particular dataset, if several datasets are used. If this list
        contains only one implementation method, it should be used for all
        datasets.
        """
        return self._detsigeff_implmethod_list
    @detsigeff_implmethod_list.setter
    def detsigeff_implmethod_list(self, methods):
        if(isinstance(methods, DetSigEffImplMethod)):
            methods = [ methods ]
        if(not issequenceof(methods, DetSigEffImplMethod)):
            raise TypeError('The detsigeff_implmethod_list property must be a '
                'sequence of DetSigEffImplMethod instances!')
        self._detsigeff_implmethod_list = methods

    @property
    def sig_gen_method(self):
        """The instance of SignalGenerationMethod that implements the signal
        generation for the specific detector and source hypothesis. It can
        be None, which means, no signal can be generated. Useful for
        data unblinding and data trial generation, where no signal is
        required.
        """
        return self._sig_gen_method
    @sig_gen_method.setter
    def sig_gen_method(self, method):
        if(not isinstance(method, SignalGenerationMethod)):
            raise TypeError('The sig_gen_method property must be an instance '
                'of SignalGenerationMethod!')
        self._sig_gen_method = method

    @property
    def n_sources(self):
        """(read-only) The number of sources within this source hypothesis
        group.
        """
        return len(self._source_list)
