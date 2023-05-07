# -*- coding: utf-8 -*-

"""This module contains utility functions for the argument parser of an analysis
script.
"""

import argparse


def create_argparser(
        description=None,
        options=True):
    """Creates an argparser with the given description and adds common options
    useful for analysis scripts.

    Parameters
    ----------
    description : str | None
        The description for the argparser.
    options : bool | dict | None
        If set to None or False, no options will be added.
        If set to True, all common analysis script options will be added.
        If set to a dictionary, individual options can be turned on and off.
        See the :func:`add_argparser_options` for possible options.
        Default is ``True``.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter
    )

    if options is True:
        options = dict()

    if isinstance(options, dict):
        add_argparser_options(
            parser=parser,
            **options)

    return parser


def add_argparser_options(
        parser,
        config=True,
        data_basepath=True,
        debug_logfile=True,
        enable_tracing=True,
        n_cpu=True,
        seed=True):
    """Adds common argparser options to the given argparser instance, useful for
    analysis scripts.

    Parameters
    ----------
    parser : instance of ArgumentParser
        The instance of ArgumentParser to which options should get added.
    config : bool
        If set to ``True``, the ``--config`` option of type ``str``
        will be added.
        It specifies the configuration file.
        The default value is ``None``.
        The option destination is ``config``.
    data_basepath : bool
        If set to ``True``, the ``--data-basepath`` option of type ``str``
        will be added.
        It specifies the base path to the data samples.
        The default value is ``None``.
        The option destination is ``data_basepath``.
    debug_logfile : bool
        If set to ``True``, the ``--debug-logfile`` option of type ``str``
        will be added.
        If not ``None``, it specifies the log file for dubug information.
        The default value is ``None``.
        The option destination is ``debug_logfile``.
    enable_tracing : bool
        If set to ``True``, the ``--enable-tracing`` option of type ``bool``
        will be added.
        If specified, enables the logging on the tracing level, i.e. a lot of
        DEBUG messages.
        The default value is ``False``.
        The option destination is ``enable_tracing``.
    n_cpu : bool
        If set to ``True``, the ``--n-cpu`` option of type ``int``
        will be added.
        It specifies the number of CPUs to utilize where parallelization is
        possible.
        The default value is ``1``.
        The option destination is ``n_cpu``.
    seed : bool
        If set to ``True``, the ``--seed`` option of type ``int``
        will be added.
        It specifies the seed for the random number generator.
        The default value is ``0``.
        The option destination is ``seed``.
    """
    if config:
        parser.add_argument(
            '--config',
            dest='config',
            default=None,
            type=str,
            help='The configuration file. '
                 '(default=None)')

    if data_basepath:
        parser.add_argument(
            '--data-basepath',
            dest='data_basepath',
            default=None,
            type=str,
            help='The base path to the data samples. '
                 '(default=None)')

    if debug_logfile:
        parser.add_argument(
            '--debug-logfile',
            dest='debug_logfile',
            default=None,
            type=str,
            help='If not None, it specifies the log file for dubug '
                 'information. '
                 '(default=None)')

    if enable_tracing:
        parser.add_argument(
            '--enable-tracing',
            dest='enable_tracing',
            default=False,
            action='store_true',
            help='If specified, enables the logging on the tracing level, i.e. '
                 'a lot of DEBUG messages. '
                 '(default=False)')

    if n_cpu:
        parser.add_argument(
            '--n-cpu',
            dest='n_cpu',
            default=1,
            type=int,
            help='The number of CPUs to utilize where parallelization is '
                 'possible. '
                 '(default=1)')

    if seed:
        parser.add_argument(
            '--seed',
            dest='seed',
            default=0,
            type=int,
            help='The seed for the random number generator. '
                 '(default=0)')
