# -*- coding: utf-8 -*-

import pickle

from skyllh.core.py import (
    int_cast,
)
from skyllh.cluster.srvclt import (
    Message,
    receive_object_from_socket,
)


class Command(object):
    """Base class for a command. A command has a command string plus optional
    additional data.
    """
    _cmd = ''

    def __init__(self):
        super(Command, self).__init__()

    def as_message(self):
        msg = pickle.dumps(self)
        return Message(msg)

    def send(self, sock):
        self.as_message().send(sock)

    def is_same_as(self, cmd):
        return (cmd._cmd == self._cmd)


class ACK(Command):
    _cmd = 'ACK'

    def __init__(self):
        super(ACK, self).__init__()


class MSG(Command):
    _cmd = 'MSG'

    def __init__(self, msg):
        super(MSG, self).__init__()

        self.msg = msg


class ShutdownCN(Command):
    _cmd = 'SHUTDOWNCN'

    def __init__(self):
        super(ShutdownCN, self).__init__()


class RegisterCN(Command):
    _cmd = 'REGCN'

    def __init__(self, cn_start_time, cn_live_time):
        """Creates a register compute node command.

        Parameters
        ----------
        cn_start_time : int
            The compute node's start time as unix time stamp.
        cn_live_time : int
            The compute node's live time. After that time the CN should be
            considered dead.
        """
        super(RegisterCN, self).__init__()

        self.cn_start_time = cn_start_time
        self.cn_live_time = cn_live_time

    @property
    def cn_start_time(self):
        """The CN's start time as unix time stamp.
        """
        return self._cn_start_time

    @cn_start_time.setter
    def cn_start_time(self, t):
        t = int_cast(
            t,
            'The cn_start_time property must be castable to type int!')
        self._cn_start_time = t

    @property
    def cn_live_time(self):
        """The CN's live time in seconds.
        """
        return self._cn_live_time

    @cn_live_time.setter
    def cn_live_time(self, t):
        t = int_cast(
            t,
            'The cn_live_time property must be castable to type int!')
        self._cn_live_time = t


def receive_command_from_socket(sock, blocksize=2048):
    """Receives a command from the given socket.
    """
    return receive_object_from_socket(sock, blocksize=blocksize)
