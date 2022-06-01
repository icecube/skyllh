# -*- coding: utf-8 -*-

"""The trialdata module of SkyLLH provides a trial data manager class that
manages the data of an analysis trial. It provides also possible additional data
fields and their calculation, which are required by the particular analysis.
The rational behind this manager is to compute data fields only once, which can
then be used by different analysis objects, like PDF objects.
"""

from collections import OrderedDict
import numpy as np

from skyllh.core.debugging import get_logger
from skyllh.core import display as dsp
from skyllh.core.py import (
    classname,
    func_has_n_args,
    int_cast,
    issequenceof,
    typename
)
from skyllh.core.storage import DataFieldRecordArray


logger = get_logger(__name__)


class DataField(object):
    """This class defines a data field and its calculation that is used by an
    Analysis class instance. The calculation is defined through an external
    function.
    """
    def __init__(
            self, name, func, fitparam_names=None, dt=None):
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
            `__call__(tdm, src_hypo_group_manager, fitparams)`,
            where `tdm` is the TrialDataManager instance holding the event data,
            `src_hypo_group_manager` is the SourceHypoGroupManager instance, and
            `fitparams` is the dictionary with the current fit parameter names
            and values. If the data field depends solely on source parameters,
            the call signature must be `__call__(tdm, src_hypo_group_manager)`
            instead.
        fitparam_names : sequence of str | None
            The sequence of str instances specifying the names of the fit
            parameters this data field depends on. If set to None, the data
            field does not depend on any fit parameters.
        dt : numpy dtype | str | None
            If specified it defines the data type this data field should have.
            If a str instance is given, it defines the name of the data field
            whose data type should be taken for this data field.
        """
        super(DataField, self).__init__()

        self.name = name
        self.func = func

        if(fitparam_names is None):
            fitparam_names = []
        if(not issequenceof(fitparam_names, str)):
            raise TypeError('The fitparam_names argument must be None or a '
                'sequence of str instances!')
        self._fitparam_name_list = list(fitparam_names)

        self.dt = dt

        # Define the list of fit parameter values for which the fit parameter
        # depend data field values have been calculated for.
        self._fitparam_value_list = [None]*len(self._fitparam_name_list)

        # Define the member variable that holds the numpy ndarray with the data
        # field values.
        self._values = None

        # Define the most efficient `calculate` method for this kind of data
        # field.
        if(func_has_n_args(self._func, 2)):
            self.calculate = self._calc_source_values
        elif(len(self._fitparam_name_list) == 0):
            self.calculate = self._calc_static_values
        else:
            self.calculate = self._calc_fitparam_dependent_values

    @property
    def name(self):
        """The name of the data field.
        """
        return self._name
    @name.setter
    def name(self, name):
        if(not isinstance(name, str)):
            raise TypeError('The name property must be an instance of str!')
        self._name = name

    @property
    def func(self):
        """The function that calculates the data field values.
        """
        return self._func
    @func.setter
    def func(self, f):
        if(not callable(f)):
            raise TypeError('The func property must be a callable object!')
        if((not func_has_n_args(f, 2)) and
           (not func_has_n_args(f, 3))):
            raise TypeError('The func property must be a function with 2 or 3 '
                'arguments!')
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
        if(obj is not None):
            if((not isinstance(obj, np.dtype)) and
               (not isinstance(obj, str))):
                raise TypeError(
                    'The dt property must be None, an instance of numpy.dtype, '
                    'or an instance of str! Currently it is of type %s.'%(
                        str(type(obj))))
        self._dt = obj

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
        if(self._values is not None):
            dtype = str(self._values.dtype)
            try:
                vmin = np.min(self._values)
            except:
                pass
            try:
                vmax = np.max(self._values)
            except:
                pass
        s = '{}: {}: '.format(classname(self), self.name)
        s +='{dtype: '
        s += '{}, vmin: {: .3e}, vmax: {: .3e}'.format(
            dtype, vmin, vmax)
        s += '}'

        return s

    def _get_desired_dtype(self, tdm):
        """Retrieves the data type this field should have. It's None, if no
        data type was defined for this data field.
        """
        if(self._dt is not None):
            if(isinstance(self._dt, str)):
                self._dt = tdm.get_dtype(self._dt)
        return self._dt

    def _convert_to_desired_dtype(self, tdm, values):
        """Converts the data type of the given values array to the given data
        type.
        """
        dt = self._get_desired_dtype(tdm)
        if(dt is not None):
            values = values.astype(dt, copy=False)
        return values

    def _calc_source_values(
            self, tdm, src_hypo_group_manager, fitparams):
        """Calculates the data field values utilizing the defined external
        function. The data field values solely depend on fixed source
        parameters.
        """
        self._values = self._func(tdm, src_hypo_group_manager)
        if(not isinstance(self._values, np.ndarray)):
            raise TypeError(
                'The calculation function for the data field "%s" must '
                'return an instance of numpy.ndarray! '
                'Currently it is of type "%s".'%(
                    self._name, typename(type(self._values))))

        # Convert the data type.
        self._values = self._convert_to_desired_dtype(tdm, self._values)

    def _calc_static_values(
            self, tdm, src_hypo_group_manager, fitparams):
        """Calculates the data field values utilizing the defined external
        function, that are static and only depend on source parameters.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance this data field is part of and is
            holding the event data.
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        fitparams : dict
            The dictionary holding the current fit parameter names and values.
            By definition this dictionary is empty.
        """
        values = self._func(tdm, src_hypo_group_manager, fitparams)
        if(not isinstance(values, np.ndarray)):
            raise TypeError(
                'The calculation function for the data field "%s" must '
                'return an instance of numpy.ndarray! '
                'Currently it is of type "%s".'%(
                    self._name, typename(type(values))))

        # Convert the data type.
        values = self._convert_to_desired_dtype(tdm, values)

        # Set the data values. This will add the data field to the
        # DataFieldRecordArray if it does not exist yet.
        tdm.events[self._name] = values

    def _calc_fitparam_dependent_values(
            self, tdm, src_hypo_group_manager, fitparams):
        """Calculate data field values utilizing the defined external
        function, that depend on fit parameter values. We check if the fit
        parameter values have changed.

        Parameters
        ----------
        tdm : instance of TrialDataManager
            The TrialDataManager instance this data field is part of and is
            holding the event data.
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        fitparams : dict
            The dictionary holding the current fit parameter names and values.
        """
        if(self._name not in tdm.events):
            # It's the first time this method is called, so we need to calculate
            # the data field values for sure.
            values = self._func(tdm, src_hypo_group_manager, fitparams)
            if(not isinstance(values, np.ndarray)):
                raise TypeError(
                    'The calculation function for the data field "%s" must '
                    'return an instance of numpy.ndarray! '
                    'Currently it is of type "%s".'%(
                        self._name, typename(type(values))))

            # Convert the data type.
            values = self._convert_to_desired_dtype(tdm, values)

            # Set the data values. This will add the data field to the
            # DataFieldRecordArray if it does not exist yet.
            tdm.events[self._name] = values

            # We store the fit parameter values for which the field values were
            # calculated for. So they have to get recalculated only when the
            # fit parameter values the field depends on change.
            self._fitparam_value_list = [
                fitparams[name] for name in self._fitparam_name_list
            ]

            return

        for (idx, fitparam_name) in enumerate(self._fitparam_name_list):
            if(fitparams[fitparam_name] != self._fitparam_value_list[idx]):
                # This current fit parameter value has changed. So we need to
                # re-calculate the data field values.
                values = self._func(tdm, src_hypo_group_manager, fitparams)

                # Convert the data type.
                values = self._convert_to_desired_dtype(tdm, values)

                # Set the data values.
                tdm.events[self._name] = values

                # Store the new fit parameter values.
                self._fitparam_value_list = [
                    fitparams[name] for name in self._fitparam_name_list
                ]

                break


class TrialDataManager(object):
    """The TrialDataManager class manages the event data for an analysis trial.
    It provides possible additional data fields and their calculation.
    New data fields can be defined via the `add_data_field` method.
    Whenever a new trial is being initialized the data fields get re-calculated.
    The data trial manager is provided to the PDF evaluation method.
    Hence, data fields are calculated only once.
    """
    def __init__(self, index_field_name=None):
        """Creates a new TrialDataManager instance.

        Parameters
        ----------
        index_field_name : str | None
            The name of the field that should be used as primary index field.
            If provided, the events will be sorted along this data field. This
            might be useful for run-time performance.
        """
        super(TrialDataManager, self).__init__()

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

        # Define the list of data fields that depend on fit parameters. These
        # data fields have to be re-calculated whenever a fit parameter value
        # changes.
        self._fitparam_data_fields_dict = OrderedDict()

        # Define the member variable that will hold the raw events for which the
        # data fields get calculated.
        self._events = None

        # Define the member variable that holds the source to event index
        # mapping.
        self._src_ev_idxs = None

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
        if(name is not None):
            if(not isinstance(name, str)):
                raise TypeError(
                    'The index_field_name property must be an instance of '
                    'type str!')
        self._index_field_name = name

    @property
    def events(self):
        """The DataFieldRecordArray instance holding the data events, which
        should get evaluated.
        """
        return self._events
    @events.setter
    def events(self, arr):
        if(not isinstance(arr, DataFieldRecordArray)):
            raise TypeError(
                'The events property must be an instance of '
                'DataFieldRecordArray!')
        self._events = arr

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
    def src_ev_idxs(self):
        """(read-only) The 2-tuple holding the source index and event index
        1d ndarray arrays.
        """
        return self._src_ev_idxs

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
        if((self._events is not None) and
           (name in self._events.field_name_list)):
            return True

        # Check if the data field is a user defined data field.
        if((name in self._source_data_fields_dict) or
           (name in self._pre_evt_sel_static_data_fields_dict) or
           (name in self._static_data_fields_dict) or
           (name in self._fitparam_data_fields_dict)):
            return True

        return False

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
        s2 = ''
        for (idx, dfield) in enumerate(self._source_data_fields_dict):
            if(idx > 0):
                s2 += '\n'
            s2 += str(dfield)
        if(s2 == ''):
            s2 = 'None'
        s1 += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s2)
        s += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s1)
        s += '\n'

        s1 = 'Pre-event-selection static data fields:\n'
        s2 = ''
        for (idx, dfield) in enumerate(self._pre_evt_sel_static_data_fields_dict):
            if(idx > 0):
                s2 += '\n'
            s2 += str(dfield)
        s1 += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s2)
        s += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s1)
        s += '\n'

        s1 = 'Static data fields:\n'
        s2 = ''
        for (idx, dfield) in enumerate(self._static_data_fields_dict):
            if(idx > 0):
                s2 += '\n'
            s2 += str(dfield)
        s1 += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s2)
        s += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s1)
        s += '\n'

        s1 = 'Fitparam data fields:\n'
        s2 = ''
        for (idx, dfield) in enumerate(self._fitparam_data_fields_dict):
            if(idx > 0):
                s2 += '\n'
            s2 += str(dfield)
        if(s2 == ''):
            s2 = 'None'
        s1 += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s2)
        s += dsp.add_leading_text_line_padding(dsp.INDENTATION_WIDTH, s1)

        return s

    def change_source_hypo_group_manager(self, src_hypo_group_manager):
        """Recalculate the source data fields.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The SourceHypoGroupManager manager that defines the groups of
            source hypotheses.
        """
        self.calculate_source_data_fields(src_hypo_group_manager)

    def initialize_trial(
            self, src_hypo_group_manager, events, n_events=None,
            evt_sel_method=None, store_src_ev_idxs=False, tl=None):
        """Initializes the trial data manager for a new trial. It sets the raw
        events, calculates pre-event-selection data fields, performs a possible
        event selection and calculates the static data fields for the left-over
        events.

        Parameters
        ----------
        src_hypo_group_manager : SourceHypoGroupManager instance
            The instance of SourceHypoGroupManager that defines the source
            hypothesis groups.
        events : DataFieldRecordArray instance
            The DataFieldRecordArray instance holding the entire raw events.
        n_events : int | None
            The total number of events of the data set this trial data manager
            corresponds to.
            If None, the number of events is taken from the number of events
            present in the ``events`` array.
        evt_sel_method : EventSelectionMethod | None
            The optional event selection method that should be used to select
            potential signal events.
        store_src_ev_idxs : bool
            If the evt_sel_method is not None, it determines if source and
            event indices of the selected events should get calculated and
            stored.
        tl : TimeLord | None
            The optional TimeLord instance that should be used for timing
            measurements.
        """
        # Set the events property, so that the calculation functions of the data
        # fields can access them.
        self.events = events

        if(n_events is None):
            n_events = len(self._events)
        self.n_events = n_events

        # Calculate pre-event-selection data fields that are required by the
        # event selection method.
        self.calculate_pre_evt_sel_static_data_fields(src_hypo_group_manager)

        if(evt_sel_method is not None):
            logger.debug(
                f'Performing event selection method '
                f'"{classname(evt_sel_method)}".')
            (selected_events, src_ev_idxs) = evt_sel_method.select_events(
                self._events, tl=tl, ret_src_ev_idxs=store_src_ev_idxs)
            logger.debug(
                f'Selected {len(selected_events)} out of {len(self._events)} '
                'events.')
            self.events = selected_events
            self._src_ev_idxs = src_ev_idxs

        # Sort the events by the index field, if a field was provided.
        if(self._index_field_name is not None):
            logger.debug(
                'Sorting events in index field "{}"'.format(
                    self._index_field_name))
            self._events.sort_by_field(self._index_field_name)

        # Now calculate all the static data fields. This will increment the
        # trial data state ID.
        self.calculate_static_data_fields(src_hypo_group_manager)

    def add_source_data_field(self, name, func, dt=None):
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
            `__call__(tdm, src_hypo_group_manager, fitparams)`, where
            `tdm` is the TrialDataManager instance holding the event data,
            `src_hypo_group_manager` is the SourceHypoGroupManager instance,
            and `fitparams` is an unused interface argument.
        dt : numpy dtype | str | None
            If specified it defines the data type this data field should have.
            If a str instance is given, it defines the name of the data field
            whose data type should be taken for the data field.
        """
        if(name in self):
            raise KeyError(
                'The data field "%s" is already defined!'%(name))

        data_field = DataField(name, func, dt=dt)

        self._source_data_fields_dict[name] = data_field

    def add_data_field(
            self, name, func, fitparam_names=None, dt=None, pre_evt_sel=False):
        """Adds a new data field to the manager.

        Parameters
        ----------
        name : str
            The name of the data field. It serves as the identifier for the
            data field.
        func : callable
            The function that calculates the data field values. The call
            signature must be
            `__call__(tdm, src_hypo_group_manager, fitparams)`, where
            `tdm` is the TrialDataManager instance holding the event data,
            `src_hypo_group_manager` is the SourceHypoGroupManager instance,
            and `fitparams` is the dictionary with the current fit parameter
            names and values.
        fitparam_names : sequence of str | None
            The sequence of str instances specifying the names of the fit
            parameters this data field depends on. If set to None, it means that
            the data field does not depend on any fit parameters.
        dt : numpy dtype | str | None
            If specified it defines the data type this data field should have.
            If a str instance is given, it defines the name of the data field
            whose data type should be taken for the data field.
        pre_evt_sel : bool
            Flag if this data field should get calculated before potential
            signal events get selected (True), or afterwards (False).
            Default is False.
        """
        if(name in self):
            raise KeyError(
                'The data field "%s" is already defined!'%(name))

        if(pre_evt_sel and (fitparam_names is not None)):
            raise ValueError(
                f'The pre-event-selection data field "{name}" must not depend '
                 'on fit parameters!')

        data_field = DataField(name, func, fitparam_names, dt=dt)

        if(pre_evt_sel):
            self._pre_evt_sel_static_data_fields_dict[name] = data_field
        elif(fitparam_names is None):
            self._static_data_fields_dict[name] = data_field
        else:
            self._fitparam_data_fields_dict[name] = data_field

    def calculate_source_data_fields(self, src_hypo_group_manager):
        """Calculates the data values of the data fields that solely depend on
        source parameters.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        """
        if(len(self._source_data_fields_dict) == 0):
            return

        fitparams = None
        for (name, dfield) in self._source_data_fields_dict.items():
            dfield.calculate(self, src_hypo_group_manager, fitparams)

        self._trial_data_state_id += 1

    def calculate_pre_evt_sel_static_data_fields(self, src_hypo_group_manager):
        """Calculates the data values of the data fields that should be
        available for the event selection method and do not depend on any fit
        parameters.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        """
        if(len(self._pre_evt_sel_static_data_fields_dict) == 0):
            return

        fitparams = dict()
        for (name, dfield) in self._pre_evt_sel_static_data_fields_dict.items():
            dfield.calculate(self, src_hypo_group_manager, fitparams)

        self._trial_data_state_id += 1

    def calculate_static_data_fields(self, src_hypo_group_manager):
        """Calculates the data values of the data fields that do not depend on
        any source or fit parameters.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        """
        if(len(self._static_data_fields_dict) == 0):
            return

        fitparams = dict()
        for (name, dfield) in self._static_data_fields_dict.items():
            dfield.calculate(self, src_hypo_group_manager, fitparams)

        self._trial_data_state_id += 1

    def calculate_fitparam_data_fields(self, src_hypo_group_manager, fitparams):
        """Calculates the data values of the data fields that depend on fit
        parameter values.

        Parameters
        ----------
        src_hypo_group_manager : instance of SourceHypoGroupManager
            The instance of SourceHypoGroupManager, which defines the groups of
            source hypotheses.
        fitparams : dict
            The dictionary holding the fit parameter names and values.
        """
        if(len(self._fitparam_data_fields_dict) == 0):
            return

        for (name, dfield) in self._fitparam_data_fields_dict.items():
            dfield.calculate(self, src_hypo_group_manager, fitparams)

        self._trial_data_state_id += 1

    def get_data(self, name):
        """Gets the data for the given data field name. The data is stored
        either in the raw events record ndarray or in one of the additional
        defined data fields. Data from the raw events record ndarray is
        prefered.

        Parameters
        ----------
        name : str
            The name of the data field for which to retrieve the data.

        Returns
        -------
        data : numpy ndarray
            The data of the requested data field.

        Raises
        ------
        KeyError
            If the given data field is not defined.
        """
        if((self._events is not None) and
           (name in self._events.field_name_list)):
            return self._events[name]

        if(name in self._source_data_fields_dict):
            data = self._source_data_fields_dict[name].values

            # Broadcast the value of an one-element 1D ndarray to the length
            # of the number of events. Note: Make sure that we don't broadcast
            # recarrays.
            if(self._events is not None):
                if((len(data) == 1) and (data.ndim == 1) and
                   (data.dtype.fields is None)):
                    data = np.repeat(data, len(self._events))
        else:
            raise KeyError(
                f'The data field "{name}" is not defined!')

        return data

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
