# -*- coding: utf-8 -*-

"""The trialdata module of SkyLLH provides a trial data manager class that
manages the data of an analysis trial. It provides also possible additional data
fields and their calculation, which are required by the particular analysis.
The rational behind this manager is to compute data fields only once, which can
then be used by different analysis objects, like PDF objects.
"""

from collections import OrderedDict
import numpy as np

from skyllh.core.debugging import (
    get_logger,
)
from skyllh.core import display as dsp
from skyllh.core.py import (
    classname,
    func_has_n_args,
    int_cast,
    issequenceof,
)
from skyllh.core.storage import (
    DataFieldRecordArray,
)


logger = get_logger(__name__)


class DataField(object):
    """This class defines a data field and its calculation that is used by an
    Analysis class instance. The calculation is defined through an external
    function.
    """
    def __init__(
            self,
            name,
            func,
            global_fitparam_names=None,
            dt=None,
            is_src_field=False,
            is_srcevt_data=False,
            **kwargs):
        """Creates a new instance of DataField that might depend on fit
        parameters.

        Parameters
        ----------
        name : str
            The name of the data field. It serves as the identifier for the
            data field.
        func : callable
            The function that calculates the values of this data field. The call
            signature must be

                __call__(tdm, shg_mgr, pmm, global_fitparams_dict=None)

            where ``tdm`` is the instance of TrialDataManager holding the trial
            event data, ``shg_mgr`` is the instance of SourceHypoGroupManager,
            ``pmm`` is the instance of ParameterModelMapper, and
            ``global_fitparams_dict`` is the dictionary with the current global
            fit parameter names and values.
        global_fitparam_names : str | sequence of str | None
            The sequence of str instances specifying the names of the global fit
            parameters this data field depends on. If set to None, the data
            field does not depend on any fit parameters.
        dt : numpy dtype | str | None
            If specified it defines the data type this data field should have.
            If a str instance is given, it defines the name of the data field
            whose data type should be taken for this data field.
        is_src_field : bool
            Flag if this data field is a source data field (``True``) and values
            should be stored within this DataField instance, instead of the
            events DataFieldRecordArray instance of the TrialDataManager
            (``False``).
        is_srcevt_data : bool
            Flag if the data field will hold source-event data, i.e. data of
            length N_values. In that case the data cannot be stored within the
            events attribute of the TrialDataManager, but must be stored in the
            values attribute of this DataField instance.
        """
        super().__init__(**kwargs)

        self.name = name
        self.func = func

        if global_fitparam_names is None:
            global_fitparam_names = []
        if isinstance(global_fitparam_names, str):
            global_fitparam_names = [global_fitparam_names]
        if not issequenceof(global_fitparam_names, str):
            raise TypeError(
                'The global_fitparam_names argument must be None or a sequence '
                'of str instances! It is of type '
                f'{classname(global_fitparam_names)}!')
        self._global_fitparam_name_list = list(global_fitparam_names)

        self.dt = dt

        # Define the list of fit parameter values for which the fit parameter
        # depend data field values have been calculated for.
        self._global_fitparam_value_list = [None] *\
            len(self._global_fitparam_name_list)

        self._is_srcevt_data = is_srcevt_data

        # Define the member variable that holds the numpy ndarray with the data
        # field values.
        self._values = None

        # Define the most efficient `calculate` method for this kind of data
        # field.
        if is_src_field:
            self.calculate = self._calc_source_values
        elif len(self._global_fitparam_name_list) == 0:
            self.calculate = self._calc_static_values
        else:
            self.calculate = self._calc_global_fitparam_dependent_values

    @property
    def name(self):
        """The name of the data field.
        """
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError(
                'The name property must be an instance of str!'
                f'It is of type {classname(name)}!')
        self._name = name

    @property
    def func(self):
        """The function that calculates the data field values.
        """
        return self._func

    @func.setter
    def func(self, f):
        if not callable(f):
            raise TypeError(
                'The func property must be a callable object!')
        if (not func_has_n_args(f, 3)) and\
           (not func_has_n_args(f, 4)):
            raise TypeError(
                'The func property must be a function with 3 or 4 arguments!')
        self._func = f

    @property
    def dt(self):
        """The numpy dtype object defining the data type of this data field.
        A str instance defines the name of the data field whose data type should
        be taken for this data field.
        It is None, if there is no explicit data type defined for this data
        field.
        """
        return self._dt

    @dt.setter
    def dt(self, obj):
        if obj is not None:
            if (not isinstance(obj, np.dtype)) and\
               (not isinstance(obj, str)):
                raise TypeError(
                    'The dt property must be None, an instance of numpy.dtype, '
                    'or an instance of str! Currently it is of type '
                    f'{classname(obj)}.')
        self._dt = obj

    @property
    def is_srcevt_data(self):
        """(read-only) Flag if the data field contains source-event data, i.e.
        is of length N_values.
        """
        return self._is_srcevt_data

    @property
    def values(self):
        """(read-only) The calculated data values of the data field.
        """
        return self._values

    def __str__(self):
        """Pretty string representation of this DataField instance.
        """
        dtype = 'None'
        vmin = np.nan
        vmax = np.nan

        if self._values is not None:
            dtype = str(self._values.dtype)
            vmin = np.min(self._values)
            vmax = np.max(self._values)

        s = f'{classname(self)}: {self.name}: '
        s += '{dtype: '
        s += f'{dtype}, vmin: {vmin: .3e}, vmax: {vmax: .3e}'
        s += '}'

        return s

    def _get_desired_dtype(self, tdm):
        """Retrieves the data type this field should have. It's ``None``, if no
        data type was defined for this data field.
        """
        if self._dt is not None:
            if isinstance(self._dt, str):
                # The _dt attribute defines the name of the data field whose
                # data type should be used.
                self._dt = tdm.get_dtype(self._dt)
        return self._dt

    def _convert_to_desired_dtype(self, tdm, values):
        """Converts the data type of the given values array to the given data
        type.
        """
        dt = self._get_desired_dtype(tdm)
        if dt is not None:
            values = values.astype(dt, copy=False)
        return values

    def _calc_source_values(
            self,
            tdm,
            shg_mgr,
            pmm):
        """Calculates the data field values utilizing the defined external
        function. The data field values solely depend on fixed source
        parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance this data field is part of and is
            holding the event data.
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the source
            hypothesis groups.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper, which defines the global
            parameters and their mapping to local source parameters.
        """
        self._values = self._func(
            tdm=tdm,
            shg_mgr=shg_mgr,
            pmm=pmm)

        if not isinstance(self._values, np.ndarray):
            raise TypeError(
                f'The calculation function for the data field "{self._name}" '
                'must return an instance of numpy.ndarray! '
                f'Currently it is of type "{classname(self._values)}".')

        # Convert the data type.
        self._values = self._convert_to_desired_dtype(tdm, self._values)

    def _calc_static_values(
            self,
            tdm,
            shg_mgr,
            pmm):
        """Calculates the data field values utilizing the defined external
        function, that are static and only depend on source parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance this data field is part of and is
            holding the event data.
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the source
            hypothesis groups.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper, which defines the global
            parameters and their mapping to local source parameters.
        """
        values = self._func(
            tdm=tdm,
            shg_mgr=shg_mgr,
            pmm=pmm)

        if not isinstance(values, np.ndarray):
            raise TypeError(
                f'The calculation function for the data field "{self._name}" '
                'must return an instance of numpy.ndarray! '
                f'Currently it is of type "{classname(values)}".')

        # Convert the data type.
        values = self._convert_to_desired_dtype(tdm, values)

        if self._is_srcevt_data:
            n_values = tdm.get_n_values()
            if values.shape[0] != n_values:
                raise ValueError(
                    'The calculation function for the data field '
                    f'"{self._name}" must return a numpy ndarray of shape '
                    f'({n_values},), but the shape is {values.shape}!')
            self._values = values
        else:
            # Set the data values. This will add the data field to the
            # DataFieldRecordArray if it does not exist yet.
            tdm.events[self._name] = values

    def _calc_global_fitparam_dependent_values(
            self,
            tdm,
            shg_mgr,
            pmm,
            global_fitparams_dict):
        """Calculate data field values utilizing the defined external
        function, that depend on fit parameter values. We check if the fit
        parameter values have changed.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance this data field is part of and is
            holding the event data.
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the source
            hypothesis groups.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper defining the mapping of the
            global parameters to local source parameters.
        global_fitparams_dict : dict
            The dictionary holding the current global fit parameter names and
            values.
        """
        # Determine if we need to calculate the values.
        calc_values = False

        if self._name not in tdm.events:
            calc_values = True
        else:
            for (idx, name) in enumerate(self._global_fitparam_name_list):
                if global_fitparams_dict[name] !=\
                   self._global_fitparam_value_list[idx]:
                    calc_values = True
                    break

        if not calc_values:
            return

        values = self._func(
            tdm=tdm,
            shg_mgr=shg_mgr,
            pmm=pmm,
            global_fitparams_dict=global_fitparams_dict)

        if not isinstance(values, np.ndarray):
            raise TypeError(
                'The calculation function for the data field '
                f'"{self._name}" must return an instance of numpy.ndarray! '
                f'Currently it is of type "{classname(values)}".')

        # Convert the data type.
        values = self._convert_to_desired_dtype(tdm, values)

        if self._is_srcevt_data:
            n_values = tdm.get_n_values()
            if values.shape[0] != n_values:
                raise ValueError(
                    'The calculation function for the data field '
                    f'"{self._name}" must return a numpy ndarray of shape '
                    f'({n_values},), but the shape is {values.shape}!')
            self._values = values
        else:
            # Set the data values. This will add the data field to the
            # DataFieldRecordArray if it does not exist yet.
            tdm.events[self._name] = values

        # We store the global fit parameter values for which the field values
        # were calculated. So they have to get recalculated only when the
        # global fit parameter values, the field depends on, change.
        self._global_fitparam_value_list = [
            global_fitparams_dict[name]
            for name in self._global_fitparam_name_list
        ]


class TrialDataManager(object):
    """The TrialDataManager class manages the event data for an analysis trial.
    It provides possible additional data fields and their calculation.
    New data fields can be defined via the :py:meth:`add_data_field` method.
    Whenever a new trial is being initialized the data fields get re-calculated.
    The data trial manager is provided to the PDF evaluation method.
    Hence, data fields are calculated only once.
    """
    def __init__(self, index_field_name=None, **kwargs):
        """Creates a new TrialDataManager instance.

        Parameters
        ----------
        index_field_name : str | None
            The name of the field that should be used as primary index field.
            If provided, the events will be sorted along this data field. This
            might be useful for run-time performance.
        """
        super().__init__(**kwargs)

        self.index_field_name = index_field_name

        # Define the list of data fields that depend only on the source
        # parameters.
        self._source_data_fields_dict = OrderedDict()

        # Define the list of data fields that are static and should be
        # calculated prior to a possible event selection.
        self._pre_evt_sel_static_data_fields_dict = OrderedDict()

        # Define the list of data fields that are static, i.e. don't depend on
        # any fit parameters. These fields have to be calculated only once when
        # a new evaluation data is available.
        self._static_data_fields_dict = OrderedDict()

        # Define the list of data fields that depend on global fit parameters.
        # These data fields have to be re-calculated whenever a global fit
        # parameter value changes.
        self._global_fitparam_data_fields_dict = OrderedDict()

        # Define the member variable that will hold the number of sources.
        self._n_sources = None

        # Define the member variable that will hold the total number of events
        # of the dataset this TrialDataManager belongs to.
        self._n_events = None

        # Define the member variable that will hold the raw events for which the
        # data fields get calculated.
        self._events = None

        # Define the member variable that holds the source to event index
        # mapping.
        self._src_evt_idxs = None

        # We store an integer number for the trial data state and increase it
        # whenever the state of the trial data changed. This way other code,
        # e.g. PDFs, can determine when the data changed and internal caches
        # must be flushed.
        self._trial_data_state_id = -1

    @property
    def index_field_name(self):
        """The name of the primary index data field. If not None, events will
        be sorted by this data field.
        """
        return self._index_field_name

    @index_field_name.setter
    def index_field_name(self, name):
        if name is not None:
            if not isinstance(name, str):
                raise TypeError(
                    'The index_field_name property must be an instance of '
                    f'type str! It is of type {classname(name)}!')
        self._index_field_name = name

    @property
    def events(self):
        """The DataFieldRecordArray instance holding the data events, which
        should get evaluated.
        """
        return self._events

    @events.setter
    def events(self, arr):
        if not isinstance(arr, DataFieldRecordArray):
            raise TypeError(
                'The events property must be an instance of '
                f'DataFieldRecordArray! It is of type {classname(arr)}!')
        self._events = arr

    @property
    def has_global_fitparam_data_fields(self):
        """(read-only) ``True`` if the TrialDataManager has global fit parameter
        data fields defined, ``False`` otherwise.
        """
        return len(self._global_fitparam_data_fields_dict) > 0

    @property
    def n_sources(self):
        """(read-only) The number of sources. This information is taken from
        the source hypo group manager when a new trial is initialized.
        """
        return self._n_sources

    @property
    def n_events(self):
        """The total number of events of the dataset this trial data manager
        corresponds to.
        """
        return self._n_events

    @n_events.setter
    def n_events(self, n):
        self._n_events = int_cast(
            n, 'The n_events property must be castable to type int!')

    @property
    def n_selected_events(self):
        """(read-only) The number of selected events which should get evaluated.
        """
        return len(self._events)

    @property
    def n_pure_bkg_events(self):
        """(read-only) The number of pure background events, which are not part
        of the trial data, but must be considered for the test-statistic value.
        It is the difference of n_events and n_selected_events.
        """
        return self._n_events - len(self._events)

    @property
    def src_evt_idxs(self):
        """(read-only) The 2-tuple holding the source indices and event indices
        1d ndarray arrays. This can be ``None``, indicating that all trial data
        events should be considered for all sources.
        """
        return self._src_evt_idxs

    @property
    def trial_data_state_id(self):
        """(read-only) The integer ID number of the trial data. This ID number
        can be used to determine when the trial data has changed its state.
        """
        return self._trial_data_state_id

    def __contains__(self, name):
        """Checks if the given data field is defined in this data field manager.

        Parameters
        ----------
        name : str
            The name of the data field.

        Returns
        -------
        check : bool
            True if the data field is defined in this data field manager,
            False otherwise.
        """
        # Check if the data field is part of the original trial data.
        if (self._events is not None) and\
           (name in self._events.field_name_list):
            return True

        # Check if the data field is a user defined data field.
        if (name in self._source_data_fields_dict) or\
           (name in self._pre_evt_sel_static_data_fields_dict) or\
           (name in self._static_data_fields_dict) or\
           (name in self._global_fitparam_data_fields_dict):
            return True

        return False

    def __getitem__(self, name):
        """Implements the evaluation of ``self[name]`` to access data fields.
        This method calls the :meth:`get_data` method of this class.
        """
        return self.get_data(name)

    def __str__(self):
        """Implements pretty string representation of this TrialDataManager
        instance.
        """
        s = classname(self)+':\n'
        s1 = 'Base data fields:\n'
        s2 = str(self._events)
        s1 += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s2)
        s += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s1)
        s += '\n'

        s1 = 'Source data fields:\n'
        s2 = '\n'.join(
            [
                str(df)
                for (_, df) in self._source_data_fields_dict.items()
            ]
        )
        if s2 == '':
            s2 = 'None'
        s1 += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s2)
        s += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s1)
        s += '\n'

        s1 = 'Pre-event-selection static data fields:\n'
        s2 = '\n'.join(
            [
                str(df)
                for (_, df) in self._pre_evt_sel_static_data_fields_dict.items()
            ]
        )
        if s2 == '':
            s2 = 'None'
        s1 += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s2)
        s += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s1)
        s += '\n'

        s1 = 'Static data fields:\n'
        s2 = '\n'.join(
            [
                str(df)
                for (_, df) in self._static_data_fields_dict.items()
            ]
        )
        if s2 == '':
            s2 = 'None'
        s1 += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s2)
        s += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s1)
        s += '\n'

        s1 = 'Global fitparam data fields:\n'
        s2 = '\n'.join(
            [
                str(df)
                for (_, df) in self._global_fitparam_data_fields_dict.items()
            ]
        )
        if s2 == '':
            s2 = 'None'
        s1 += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s2)
        s += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s1)

        return s

    def broadcast_sources_array_to_values_array(
            self,
            arr):
        """Broadcasts the given 1d numpy ndarray of length 1 or N_sources to a
        numpy ndarray of length N_values.

        Parameters
        ----------
        arr : instance of ndarray
            The (N_sources,)- or (1,)-shaped numpy ndarray holding values for
            each source.

        Returns
        -------
        out_arr : instance of ndarray
            The (N_values,)-shaped numpy ndarray holding the source values
            broadcasted to each event value.
        """
        arr_dtype = arr.dtype
        n_values = self.get_n_values()

        if len(arr) == 1:
            return np.full((n_values,), arr[0], dtype=arr_dtype)

        if len(arr) != self.n_sources:
            raise ValueError(
                f'The length of arr ({len(arr)}) must be 1 or equal to the '
                f'number of sources ({self.n_sources})!')

        out_arr = np.empty(
            (n_values,),
            dtype=arr.dtype)

        src_idxs = self.src_evt_idxs[0]
        v_start = 0
        for (src_idx, src_value) in enumerate(arr):
            n = np.count_nonzero(src_idxs == src_idx)
            # n = len(evt_idxs[src_idxs == src_idx])
            out_arr[v_start:v_start+n] = np.full(
                (n,), src_value, dtype=arr_dtype)
            v_start += n

        return out_arr

    def broadcast_sources_arrays_to_values_arrays(
            self,
            arrays):
        """Broadcasts the 1d numpy ndarrays to the values array.

        Parameters
        ----------
        arrays : sequence of numpy 1d ndarrays
            The sequence of (N_sources,)-shaped numpy ndarrays holding the
            parameter values.

        Returns
        -------
        out_arrays : list of numpy 1d ndarrays
            The list of (N_values,)-shaped numpy ndarrays holding the
            broadcasted array values.
        """
        out_arrays = [
            self.broadcast_sources_array_to_values_array(arr)
            for arr in arrays
        ]

        return out_arrays

    def broadcast_selected_events_arrays_to_values_arrays(
            self,
            arrays):
        """Broadcasts the given arrays of length N_selected_events to arrays
        of length N_values.

        Parameters
        ----------
        arrays : sequence of instance of ndarray
            The sequence of instance of ndarray with the arrays to be
            broadcasted.

        Returns
        -------
        out_arrays : list of instance of ndarray
            The list of broadcasted numpy ndarray instances.
        """
        evt_idxs = self._src_evt_idxs[1]
        out_arrays = [
            np.take(arr, evt_idxs)
            for arr in arrays
        ]

        return out_arrays

    def change_shg_mgr(self, shg_mgr, pmm):
        """This method is called when the source hypothesis group manager has
        changed. Hence, the source data fields need to get recalculated.

        After calling this method, a new trial should be initialized via the
        :meth:`initialize_trial` method!

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the source
            hypothesis groups.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper that defines the global
            parameters and their mapping to local source parameter.
        """
        self.calculate_source_data_fields(
            shg_mgr=shg_mgr,
            pmm=pmm)

    def initialize_trial(
            self,
            shg_mgr,
            pmm,
            events,
            n_events=None,
            evt_sel_method=None,
            tl=None):
        """Initializes the trial data manager for a new trial. It sets the raw
        events, calculates pre-event-selection data fields, performs a possible
        event selection and calculates the static data fields for the left-over
        events.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager that defines the source
            hypothesis groups.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper, that defines the global
            parameters and their mapping to local source parameters.
        events : DataFieldRecordArray instance
            The DataFieldRecordArray instance holding the entire raw events.
        n_events : int | None
            The total number of events of the data set this trial data manager
            corresponds to.
            If None, the number of events is taken from the number of events
            present in the ``events`` array.
        evt_sel_method : instance of EventSelectionMethod | None
            The optional event selection method that should be used to select
            potential signal events.
        tl : instance of TimeLord | None
            The optional TimeLord instance that should be used for timing
            measurements.
        """
        # Set the events property, so that the calculation functions of the data
        # fields can access them.
        self.events = events
        self._src_evt_idxs = None

        # Save the number of sources.
        self._n_sources = shg_mgr.n_sources

        if n_events is None:
            n_events = len(self._events)
        self.n_events = n_events

        # Calculate pre-event-selection data fields that are required by the
        # event selection method.
        self.calculate_pre_evt_sel_static_data_fields(
            shg_mgr=shg_mgr,
            pmm=pmm)

        if evt_sel_method is not None:
            logger.debug(
                f'Performing event selection method '
                f'"{classname(evt_sel_method)}".')
            (selected_events, src_evt_idxs) = evt_sel_method.select_events(
                events=self._events,
                tl=tl)
            logger.debug(
                f'Selected {len(selected_events)} out of {len(self._events)} '
                'events.')
            self.events = selected_events
            self._src_evt_idxs = src_evt_idxs

        # Sort the events by the index field, if a field was provided.
        if self._index_field_name is not None:
            logger.debug(
                f'Sorting events in index field "{self._index_field_name}"')
            sorted_idxs = self._events.sort_by_field(self._index_field_name)
            # If event indices are stored, we need to re-assign also those event
            # indices according to the new order.
            if self._src_evt_idxs is not None:
                self._src_evt_idxs[1] = np.take(
                    sorted_idxs, self._src_evt_idxs[1])

        # Create the src_evt_idxs property data in case it was not provided by
        # the event selection. In that case all events are selected for all
        # sources. This simplifies the implementations of the PDFs.
        if self._src_evt_idxs is None:
            self._src_evt_idxs = (
                np.repeat(np.arange(self.n_sources), self.n_selected_events),
                np.tile(np.arange(self.n_selected_events), self.n_sources)
            )

        # Now calculate all the static data fields. This will increment the
        # trial data state ID.
        self.calculate_static_data_fields(
            shg_mgr=shg_mgr,
            pmm=pmm)

    def get_n_values(self):
        """Returns the expected size of the values array after a PDF
        evaluation, which will include PDF values for all trial data events and
        all sources.

        Returns
        -------
        n : int
            The length of the expected values array after a PDF evaluation.
        """
        return len(self._src_evt_idxs[0])

    def get_values_mask_for_source_mask(self, src_mask):
        """Creates a boolean mask for the values array where entries belonging
        to the sources given by the source mask are selected.

        Parameters
        ----------
        src_mask : instance of numpy ndarray
            The (N_sources,)-shaped numpy ndarray holding the boolean selection
            of the sources.

        Returns
        -------
        values_mask : instance of numpy ndarray
            The (N_values,)-shaped numpy ndarray holding the boolean selection
            of the values.
        """
        tdm_src_idxs = self.src_evt_idxs[0]
        src_idxs = np.arange(self.n_sources)[src_mask]

        values_mask = np.zeros((self.get_n_values(),), dtype=np.bool_)

        def make_values_mask(src_idx):
            global values_mask
            values_mask |= tdm_src_idxs == src_idx

        np.vectorize(make_values_mask)(src_idxs)

        return values_mask

    def add_source_data_field(
            self,
            name,
            func,
            dt=None):
        """Adds a new data field to the manager. The data field must depend
        solely on source parameters.

        Parameters
        ----------
        name : str
            The name of the data field. It serves as the identifier for the
            data field.
        func : callable
            The function that calculates the data field values. The call
            signature must be

                __call__(tdm, shg_mgr, pmm)

            where ``tdm`` is the TrialDataManager instance holding the event
            data, ``shg_mgr`` is the instance of SourceHypoGroupManager,
            and ``pmm`` is the instance of ParameterModelMapper.
        dt : numpy dtype | str | None
            If specified it defines the data type this data field should have.
            If a str instance is given, it defines the name of the data field
            whose data type should be taken for the data field.
        """
        if name in self:
            raise KeyError(
                f'The data field "{name}" is already defined!')

        data_field = DataField(
            name=name,
            func=func,
            dt=dt,
            is_src_field=True)

        self._source_data_fields_dict[name] = data_field

    def add_data_field(
            self,
            name,
            func,
            global_fitparam_names=None,
            dt=None,
            pre_evt_sel=False,
            is_srcevt_data=False):
        """Adds a new data field to the manager.

        Parameters
        ----------
        name : str
            The name of the data field. It serves as the identifier for the
            data field.
        func : callable
            The function that calculates the data field values. The call
            signature must be

                __call__(tdm, shg_mgr, pmm, global_fitparams_dict=None)

            where ``tdm`` is the TrialDataManager instance holding the trial
            event data, ``shg_mgr`` is the instance of SourceHypoGroupManager,
            ``pmm`` is the instance of ParameterModelMapper, and
            ``global_fitparams_dict`` is the dictionary with the current global
            fit parameter names and values.
            The shape of the returned array must be (N_selected_events,).
        global_fitparam_names : str | sequence of str | None
            The sequence of str instances specifying the names of the global fit
            parameters this data field depends on. If set to ``None``, it means
            that the data field does not depend on any fit parameters.
        dt : numpy dtype | str | None
            If specified it defines the data type this data field should have.
            If a str instance is given, it defines the name of the data field
            whose data type should be taken for the data field.
        pre_evt_sel : bool
            Flag if this data field should get calculated before potential
            signal events get selected (True), or afterwards (False).
            Default is False.
        is_srcevt_data : bool
            Flag if this data field contains source-event data, hence the length
            of the data array will be N_values.
            Default is False.
        """
        if name in self:
            raise KeyError(
                'The data field "{name}" is already defined!')

        if pre_evt_sel:
            if global_fitparam_names is not None:
                raise ValueError(
                    f'The pre-event-selection data field "{name}" must not '
                    'depend on global fit parameters!')

            if is_srcevt_data:
                raise ValueError(
                    'By definition the pre-event-selection data field '
                    f'"{name}" cannot hold source-event data! The '
                    'is_srcevt_data argument must be set to False!')

        data_field = DataField(
            name=name,
            func=func,
            global_fitparam_names=global_fitparam_names,
            dt=dt,
            is_src_field=False,
            is_srcevt_data=is_srcevt_data)

        if pre_evt_sel:
            self._pre_evt_sel_static_data_fields_dict[name] = data_field
        elif global_fitparam_names is None:
            self._static_data_fields_dict[name] = data_field
        else:
            self._global_fitparam_data_fields_dict[name] = data_field

    def calculate_source_data_fields(
            self,
            shg_mgr,
            pmm):
        """Calculates the data values of the data fields that solely depend on
        source parameters.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper, that defines the global
            parameters and their mapping to local source parameters.
        """
        if len(self._source_data_fields_dict) == 0:
            return

        for (name, dfield) in self._source_data_fields_dict.items():
            dfield.calculate(
                tdm=self,
                shg_mgr=shg_mgr,
                pmm=pmm)

        self._trial_data_state_id += 1

    def calculate_pre_evt_sel_static_data_fields(
            self,
            shg_mgr,
            pmm):
        """Calculates the data values of the data fields that should be
        available for the event selection method and do not depend on any fit
        parameters.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper, that defines the global
            parameters and their mapping to local source parameters.
        """
        if len(self._pre_evt_sel_static_data_fields_dict) == 0:
            return

        for (name, dfield) in self._pre_evt_sel_static_data_fields_dict.items():
            dfield.calculate(
                tdm=self,
                shg_mgr=shg_mgr,
                pmm=pmm)

        self._trial_data_state_id += 1

    def calculate_static_data_fields(
            self,
            shg_mgr,
            pmm):
        """Calculates the data values of the data fields that do not depend on
        any source or fit parameters.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper, that defines the global
            parameters and their mapping to local source parameters.
        """
        if len(self._static_data_fields_dict) == 0:
            return

        for (name, dfield) in self._static_data_fields_dict.items():
            dfield.calculate(
                tdm=self,
                shg_mgr=shg_mgr,
                pmm=pmm)

        self._trial_data_state_id += 1

    def calculate_global_fitparam_data_fields(
            self,
            shg_mgr,
            pmm,
            global_fitparams_dict):
        """Calculates the data values of the data fields that depend on global
        fit parameter values.

        Parameters
        ----------
        shg_mgr : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        pmm : instance of ParameterModelMapper
            The instance of ParameterModelMapper, that defines the global
            parameters and their mapping to local source parameters.
        global_fitparams_dict : dict
            The dictionary holding the current global fit parameter names and
            values.
        """
        if len(self._global_fitparam_data_fields_dict) == 0:
            return

        for (name, dfield) in self._global_fitparam_data_fields_dict.items():
            dfield.calculate(
                tdm=self,
                shg_mgr=shg_mgr,
                pmm=pmm,
                global_fitparams_dict=global_fitparams_dict)

        self._trial_data_state_id += 1

    def get_data(self, name):
        """Gets the data for the given data field name. The data is stored
        either in the raw events DataFieldRecordArray or in one of the
        additional defined data fields. Data from the raw events
        DataFieldRecordArray is prefered.

        Parameters
        ----------
        name : str
            The name of the data field for which to retrieve the data.

        Returns
        -------
        data : instance of numpy ndarray
            The numpy ndarray holding the data of the requested data field.
            The length of the array is either N_sources, N_selected_events, or
            N_values.

        Raises
        ------
        KeyError
            If the given data field is not defined.
        """
        # Data fields which are static or depend on global fit parameters are
        # stored within the _events DataFieldRecordArray if they do not contain
        # source-event data. For all other cases, the data is stored in the
        # .values attribute of the DataField class instance.
        if self._events is not None and\
           name in self._events.field_name_list:
            return self._events[name]

        if name in self._source_data_fields_dict:
            return self._source_data_fields_dict[name].values

        if name in self._static_data_fields_dict:
            return self._static_data_fields_dict[name].values

        if name in self._global_fitparam_data_fields_dict:
            return self._global_fitparam_data_fields_dict[name].values

        raise KeyError(
            f'The data field "{name}" is not defined!')

    def get_dtype(self, name):
        """Gets the data type of the given data field.

        Parameters
        ----------
        name : str
            The name of the data field whose data type should get retrieved.

        Returns
        -------
        dt : numpy dtype
            The numpy dtype object of the given data field.

        Raises
        ------
        KeyError
            If the given data field is not defined.
        """
        dt = self.get_data(name).dtype

        return dt

    def is_event_data_field(self, name):
        """Checks if the given data field is an events data field, i.e. its
        length is N_selected_events.

        Parameters
        ----------
        name : str
            The name of the data field.

        Returns
        -------
        check : bool
            ``True`` if the given data field contains event data, ``False``
            otherwise.
        """
        if self._events is not None and\
           name in self._events.field_name_list:
            return True

        return False

    def is_source_data_field(self, name):
        """Checks if the given data field is a source data field, i.e. its
        length is N_sources.

        Parameters
        ----------
        name : str
            The name of the data field.

        Returns
        -------
        check : bool
            ``True`` if the given data field contains source data, ``False``
            otherwise.
        """
        if name in self._source_data_fields_dict:
            return True

        return False

    def is_srcevt_data_field(self, name):
        """Checks if the given data field is a source-event data field, i.e. its
        length is N_values.

        Parameters
        ----------
        name : str
            The name of the data field.

        Returns
        -------
        check : bool
            ``True`` if the given data field contains source-event data,
            ``False`` otherwise.
        """
        if name in self._static_data_fields_dict:
            return self._static_data_fields_dict[name].is_srcevt_data

        if name in self._global_fitparam_data_fields_dict:
            return self._global_fitparam_data_fields_dict[name].is_srcevt_data

        return False
