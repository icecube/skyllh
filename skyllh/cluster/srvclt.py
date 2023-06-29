# -*- coding: utf-8 -*-

import pickle

from skyllh.core.py import (
    str_cast,
)


class Message(object):
    @staticmethod
    def receive(sock, blocksize=2048, as_bytes=False):
        """Receives a message from the given socket.

        Parameters
        ----------
        blocksize : int
            The size in bytes of the block that should be read from the socket
            at once.
        as_bytes : bool
            If set to ``True`` the Message instance will contain a bytes
            message, otherwise it is converted to a str.

        Returns
        -------
        m : Message
            The Message instance created with the message read from the socket.
        """
        # Get the first 2 bytes to determine the length of the message.
        msglen = int.from_bytes(
            read_from_socket(sock, 2, blocksize=blocksize), 'little')

        # Read the message of length msglen bytes from the socket. Here, msg is
        # a bytes object.
        msg = read_from_socket(sock, msglen, blocksize=blocksize)

        if as_bytes:
            return Message(msg)

        return Message(str(msg, 'utf-8'))

    def __init__(self, msg):
        """Creates a new Message instance.

        Parameters
        ----------
        msg : str | bytes
            The message string of this Message instance.
        """
        self.msg = msg

    @property
    def msg(self):
        """The message string. This is either a bytes instance or a str
        instance.
        """
        return self._msg

    @msg.setter
    def msg(self, m):
        if not isinstance(m, bytes):
            m = str_cast(
                m,
                'The msg property must be of type bytes or castable to type '
                'str!')
        self._msg = m

    @property
    def length(self):
        """The length of the message in bytes.
        """
        return len(self.msg)

    def as_socket_msg(self):
        """Converts this message to a bytes instance that can be send through a
        socket. The first two bytes hold the length of the message.
        """
        smsg = len(self.msg).to_bytes(2, 'little')
        if isinstance(self.msg, bytes):
            smsg += self.msg
        else:
            smsg += bytes(self.msg, 'utf-8')

        return smsg

    def send(self, sock):
        send_to_socket(sock, self.as_socket_msg())


def send_to_socket(sock, msg):
    msglen = len(msg)
    n_bytes_sent = 0
    while n_bytes_sent < msglen:
        sent = sock.send(msg[n_bytes_sent:])
        if sent == 0:
            raise RuntimeError('Socket connection broken!')
        n_bytes_sent += sent


def read_from_socket(sock, size, blocksize=2048):
    """Reads ``size`` bytes from the socket ``sock``.
    """
    chunks = []
    n_bytes_recd = 0
    while (n_bytes_recd < size):
        chunk = sock.recv(min(size - n_bytes_recd, blocksize))
        if chunk == b'':
            raise RuntimeError('Socket connection broken!')
        chunks.append(chunk)
        n_bytes_recd += len(chunk)
    return b''.join(chunks)


def receive_object_from_socket(sock, blocksize=2048):
    """Receives a pickled Python object from the given socket.

    Parameters
    ----------
    sock : socket
    """
    m = Message.receive(sock, blocksize, as_bytes=True)
    obj = pickle.loads(m.msg)
    return obj
