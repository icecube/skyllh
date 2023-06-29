# -*- coding: utf-8 -*-

import logging
import socket

from skyllh.cluster.commands import (
    ACK,
    Command,
    MSG,
    ShutdownCN,
    RegisterCN,
    receive_command_from_socket,
)


class CNRegistryEntry(object):
    """This class provides an registry entry for a compute node. It holds the
    socket to the compute node.
    """
    def __init__(self, sock, addr, port, cn_start_time, cn_live_time):
        super(CNRegistryEntry, self).__init__()

        self.sock = sock
        self.addr = addr
        self.port = port
        self.cn_start_time = cn_start_time
        self.cn_live_time = cn_live_time

    def __del__(self):
        self.sock.close()

    @property
    def key(self):
        """(read-only) The CN's identification key.
        """
        return f'{self.addr:s}:{self.port:d}'

    def send_command(self, cmd):
        if not isinstance(cmd, Command):
            raise TypeError(
                'The cmd argument must be an instance of Command!')
        cmd.send(self.sock)


class MasterNode(object):
    """The MasterNode class provides an entity to run the SkyLLH program as a
    master node, where compute nodes can register to, so the master node can
    distribute work to the compute nodes. The work distribution is handled
    through the MasterNode instance.
    """
    def __init__(self):
        super(MasterNode, self).__init__()

        self.cn_registry = dict()

    @property
    def cn_registry(self):
        """The dictionary with the registered compute nodes.
        """
        return self._cn_registry

    @cn_registry.setter
    def cn_registry(self, d):
        if not isinstance(d, dict):
            raise TypeError(
                'The cn_registry property must be of type dict!')
        self._cn_registry = d

    def clear_cn_registry(self):
        # Close the sockets to all the CNs.
        for (cn_key, cn) in self.cn_registry.items():
            cn.sock.close()

        self.cn_registry = dict()

    def register_compute_nodes(self, n_cn=10, master_port=9999, blocksize=2048):
        logger = logging.getLogger(__name__)

        logger.debug(
            'Clearing the CN registry')
        self.clear_cn_registry()

        logger.debug(
            'Creating server TCP/IP socket')
        serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # bind the socket to a public host, and a well-known port
        master_hostname = socket.getfqdn(socket.gethostname())
        logger.debug(
            'Listening on %s:%d with %d simulanious allowed connections',
            master_hostname, master_port, n_cn)
        serversock.bind((master_hostname, master_port))
        serversock.listen(n_cn)

        try:
            while len(self._cn_registry) < n_cn:
                # Accept connections from the compute nodes in order to register
                # them.
                (clientsock, (addr, port)) = serversock.accept()
                logger.debug(
                    'Got inbound connection from %s:%d', addr, port)

                cmd = receive_command_from_socket(
                    clientsock, blocksize=blocksize)
                if not cmd.is_same_as(RegisterCN):
                    raise RuntimeError(
                        'The compute node provided an unknown command '
                        f'"{cmd.as_message().msg}"!')
                ACK().send(clientsock)

                cn = CNRegistryEntry(
                    clientsock, addr, port, cmd.cn_start_time, cmd.cn_live_time)
                self._cn_registry[cn.key] = cn
        finally:
            serversock.close()

    def send_request(self, msg):
        for (cn_key, cn) in self.cn_registry.items():
            cn.send_command(MSG(msg))

    def shutdown_compute_nodes(self):
        """Sends a stop command to all compute nodes.
        """
        for (cn_key, cn) in self._cn_registry.items():
            cn.send_command(ShutdownCN())
