# -*- coding: utf-8 -*-

import argparse
import socket
import time

from skyllh.cluster.commands import (
    ACK,
    MSG,
    RegisterCN,
    ShutdownCN,
    receive_command_from_socket,
)
from skyllh.core.py import (
    int_cast,
)


class ComputeNode(object):
    """The ComputeNode class provides an entity for stand-alone program running
    on a dedicated compute node host.
    """
    def __init__(self, live_time, master_addr, master_port):
        super(ComputeNode, self).__init__()

        self.live_time = live_time
        self.master_addr = master_addr
        self.master_port = master_port

        self._start_time = time.time()

        # Register the compute node to the master node.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.master_addr, self.master_port))

        # Send the register command to the master and tell .
        RegisterCN(self._start_time, self._live_time).send(self.sock)
        reply = receive_command_from_socket(self.sock)
        if not reply.is_same_as(ACK):
            raise RuntimeError(
                'The master node did not reply with an ACK command!')

        print(f'Registered to master {self._master_addr}:{self._master_port}')
        print(f'Runtime set to {self._live_time} seconds')

    def __del__(self):
        self.sock.close()

    @property
    def live_time(self):
        """The time in seconds this ComputeNode instance should be listening for
        requests.
        """
        return self._live_time

    @live_time.setter
    def live_time(self, t):
        t = int_cast(
            t,
            'The live_time property must be castable to type int!')
        self._live_time = t

    @property
    def master_addr(self):
        """The address of the SkyLLH master program.
        """
        return self._master_addr

    @master_addr.setter
    def master_addr(self, addr):
        if not isinstance(addr, str):
            raise TypeError(
                'The master_addr property must be of type str!')
        self._master_addr = addr

    @property
    def master_port(self):
        """The port number of the SkyLLH master program.
        """
        return self._master_port

    @master_port.setter
    def master_port(self, p):
        p = int_cast(
            p,
            'The master_port property must be castable to type int!')
        self._master_port = p

    def handle_requests(self):
        if time.time() > self._start_time + self._live_time:
            raise RuntimeError('Live-time already exceeded!')

        while True:
            # Receive a command.
            cmd = receive_command_from_socket(self.sock)
            if cmd.is_same_as(MSG):
                print(f'Received general message: {cmd.msg}')
            elif cmd.is_same_as(ShutdownCN):
                print('Received shutdown command. Shutting down.')
                self.sock.close()
                return
            else:
                print('Received unknown command! Ignoring.')

            if time.time() > self._start_time + self._live_time:
                print('Live-time exceeded. Shutting down.')
                return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SkyLLH Compute Node')
    parser.add_argument(
        'master_addr', type=str,
        help='The address (IP / hostname) of the SkyLLH master program.')
    parser.add_argument(
        'master_port', type=int, default=9999,
        help='The port number of the SkyLLH master program.')
    parser.add_argument(
        '--live-time', type=int, default=2*60*60,
        help='The time in seconds to run this compute node instance.')

    args = parser.parse_args()

    cn = ComputeNode(
        live_time=args.live_time,
        master_addr=args.master_addr,
        master_port=args.master_port)

    cn.handle_requests()
