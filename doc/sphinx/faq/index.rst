.. _faq_index:

********************************
Frequently Asked Questions (FAQ)
********************************

.. dropdown:: How can I change the flux model of my sources (for my signal events)?
   :icon: question

    When SkyLLH generates pseudo data for an analysis, it will generate background
    and signal events. It might be desired to change the flux model of the sources 
    to generate signal events following a different flux model as originally choosen
    when the analysis was created.

    The most general procedure to change the flux model(s) of the source(s) of 
    (only) the signal generator is to create a new signal generator with a 
    :py:class:`~skyllh.core.source_hypo_grouping.SourceHypoGroupManager` instance 
    that includes the sources with the changed flux model.

    The new signal generator instance should then be set to the
    :py:attr:`~skyllh.core.analysis.Analysis._sig_generator` attribute of the
    :py:class:`~skyllh.core.analysis.Analysis` class. However, if the analysis 
    signal generator relies on signal generators for each individual dataset, such
    dataset signal generators need to be re-created as well and set to the
    :py:attr:`~skyllh.core.analysis.Analysis._sig_generator_list` attribute of the
    :py:class:`~skyllh.core.analysis.Analysis` class.

    .. note::

        If the flux model(s) of the source(s) should be set for the analysis itself,
        i.e. also for the detector signal yield calculation, which is used to weigh
        the datasets of a multi-dataset analysis, the 
        :py:meth:`~skyllh.core.analysis.Analysis.change_shg_mgr` method should be
        called with the new 
        :py:class:`~skyllh.core.source_hypo_grouping.SourceHypoGroupManager` 
        instance. This method will call the ``change_shg_mgr`` method of all 
        analysis components.

        A re-creation of the signal generator instances is then not required.
