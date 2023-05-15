# -*- coding: utf-8 -*-

import numpy as np

from skyllh.core.dataset import (
    Dataset,
    DatasetData,
)
from skyllh.core.progressbar import (
    ProgressBar,
)
from skyllh.core.py import (
    classname,
    issequenceof,
)
from skyllh.core.source_hypo_grouping import (
    SourceHypoGroupManager,
)


class DetSigYieldService(
        object):
    """This class provides a service to build and hold detector signal yield
    instances for multiple datasets and source hypothesis groups.
    """

    def __init__(
            self,
            shg_mgr,
            dataset_list,
            data_list,
            ppbar=None,
            **kwargs):
        """Creates a new DetSigYieldService instance.
        """
        super().__init__(
            **kwargs)

        self._set_shg_mgr(shg_mgr)

        self.dataset_list = dataset_list
        self.data_list = data_list

        self._arr = self.construct_detsigyield_array(
            ppbar=ppbar)

    @property
    def shg_mgr(self):
        """(read-only) The instance of SourceHypoGroupManager providing the list
        of source hypothesis groups.
        """
        return self._shg_mgr

    @property
    def dataset_list(self):
        """The list of instance of Dataset for which the detector signal yields
        should be built.
        """
        return self._dataset_list

    @dataset_list.setter
    def dataset_list(self, datasets):
        if not issequenceof(datasets, Dataset):
            raise TypeError(
                'The dataset_list property must be a sequence of Dataset '
                'instances! '
                f'Its current type is {classname(datasets)}!')
        self._dataset_list = list(datasets)

    @property
    def data_list(self):
        """The list of instance of DatasetData for which the detector signal
        yields should be built.
        """
        return self._data_list

    @data_list.setter
    def data_list(self, datas):
        if not issequenceof(datas, DatasetData):
            raise TypeError(
                'The data_list property must be a sequence of DatasetData '
                'instances! '
                f'Its current type is {classname(datas)}!')
        self._data_list = list(datas)

    @property
    def arr(self):
        """(read-only) The (N_datasets, N_source_hypo_groups)-shaped numpy
        ndarray of object holding the constructed DetSigYield instances.
        """
        return self._arr

    def _set_shg_mgr(self, mgr):
        """Sets the internal member variable to the given instance of
        SourceHypoGroupManager.
        """
        if not isinstance(mgr, SourceHypoGroupManager):
            raise TypeError(
                'The shg_mgr argument must be an instance of '
                'SourceHypoGroupManager! '
                f'Its current type is {classname(mgr)}!')

        self._shg_mgr = mgr

    def change_shg_mgr(
            self,
            shg_mgr,
            ppbar=None,
    ):
        """Changes the instance of SourceHypoGroupManager of this service. This
        will also rebuild the detector signal yields.
        """
        self._set_shg_mgr(shg_mgr)

        self._arr = self.construct_detsigyield_array(
            ppbar=ppbar)

    def construct_detsigyield_array(
            self,
            ppbar=None,
    ):
        """Creates a (N_datasets, N_source_hypo_groups)-shaped numpy ndarray of
        object holding the constructed DetSigYield instances.

        Parameters
        ----------
        ppbar : instance of ProgressBar | None
            The instance of ProgressBar of the optional parent progress bar.

        Returns
        -------
        detsigyield_arr : instance of numpy ndarray
            The (N_datasets, N_source_hypo_groups)-shaped numpy ndarray of
            object holding the constructed DetSigYield instances.
        """
        n_datasets = len(self._dataset_list)

        detsigyield_arr = np.empty(
            (n_datasets,
             self._shg_mgr.n_src_hypo_groups),
            dtype=object
        )

        pbar = ProgressBar(
            self._shg_mgr.n_src_hypo_groups * n_datasets,
            parent=ppbar).start()
        for (g, shg) in enumerate(self._shg_mgr.shg_list):
            fluxmodel = shg.fluxmodel
            detsigyield_builder_list = shg.detsigyield_builder_list

            if (len(detsigyield_builder_list) != 1) and\
               (len(detsigyield_builder_list) != n_datasets):
                raise ValueError(
                    'The number of detector signal yield builders '
                    f'({len(detsigyield_builder_list)}) is not 1 and does not '
                    f'match the number of datasets ({n_datasets})!')

            for (j, (dataset, data)) in enumerate(zip(self._dataset_list,
                                                      self._data_list)):
                if len(detsigyield_builder_list) == 1:
                    # Only one detsigyield builder was defined,
                    # so we use it for all datasets.
                    detsigyield_builder = detsigyield_builder_list[0]
                else:
                    detsigyield_builder = detsigyield_builder_list[j]

                detsigyield = detsigyield_builder.construct_detsigyield(
                    dataset=dataset,
                    data=data,
                    fluxmodel=fluxmodel,
                    livetime=data.livetime,
                    ppbar=pbar)
                detsigyield_arr[j, g] = detsigyield

                pbar.increment()
        pbar.finish()

        return detsigyield_arr
