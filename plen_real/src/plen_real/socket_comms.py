#!/usr/bin/env python

import socket  # for sockets
import sys  # for exit
import pickle  # for serializing data
import struct  # for unpacking binary data


class Socket:
    def __init__(self, host="raspberrypi", port=41953, buffer_size=4096):
        # Maximumm receive message size in bytes
        self.buffer_size = buffer_size
        try:
            # create an AF_INET, STREAM socket (TCP)
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error, msg:
            print('Failed to create socket. Error code: ' + str(msg[0]) +
                  ' , Error message : ' + msg[1])
            sys.exit()
        print("Socket Created!")

        self.host = host
        # https://stackoverflow.com/questions/5882247/which-port-can-i-use-for-my-socket
        self.port = port
        try:
            self.remote_ip = socket.gethostbyname(host)

        except socket.gaierror:
            # could not resolve
            print 'Hostname could not be resolved. Exiting'
            sys.exit()

        print('Ip address of ' + self.host + ' is ' + self.remote_ip)

        # Connect to remote server
        self.s.connect((self.remote_ip, self.port))

        print('Socket Connected to ' + self.host + ' on ip ' + self.remote_ip)

    def send_message(self, data):
        """ Send socket message
        """
        data_stream = pickle.dumps(data)
        self.send_one_message(data_stream)

    def send_one_message(self, data):
        """ Sends message size first,
            then sends actual message
        """
        length = len(data)
        self.s.sendall(struct.pack('!I', length))
        self.s.sendall(data)

    def receive_message(self):
        """ Receive socket message
        """
        data_stream = self.receive_buffered_message()
        data = pickle.loads(data_stream.decode())

        return data

    def receive_buffered_message(self):
        """ Receive size in bytes of message
            which will now be received.
        """
        # Buffer size will be a 4-byte message
        lengthbuf = self.recvall(4)
        length = struct.unpack('!I', lengthbuf)
        return self.recvall(length)

    def recvall(self, count):
        """ Receie as many bytes as exist in 'count'.
            Esnures that entire message is received
            if length is known and total msg size
            is below 4GB
        """
        buf = b''
        while count:
            newbuf = self.s.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf