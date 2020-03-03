#!/usr/bin/env python

import socket  # for sockets
import sys  # for exit
import pickle  # for serializing data
import struct  # for unpacking binary data
import time


class SocketClient:
    def __init__(self, host="raspberrypi", port=41953, buffer_size=1024):
        # Maximumm receive message size in bytes
        self.buffer_size = buffer_size
        try:
            # create an AF_INET, STREAM socket (TCP - STREAM | UDP - DGRAM)
            self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
        data_stream = self.s.recv(self.buffer_size)
        # data = data_stream
        data = pickle.loads(data_stream)

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


class SocketServer:
    def __init__(self, host="bot", port=41953, buffer_size=1024):
        localIP = "192.168.0.17"

        msgFromServer = pickle.dumps([0.01, 0.02, 0.03])

        bytesToSend = str.encode(msgFromServer)

        # Create a datagram socket

        self.s = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

        # Bind to address and ip

        self.s.bind((localIP, port))

        print("UDP server up and listening")

        # Listen for incoming datagrams

        while (True):

            bytesAddressPair = self.s.recvfrom(buffer_size)

            message = bytesAddressPair[0]

            address = bytesAddressPair[1]

            clientMsg = "Message from Client:{}".format(message)
            clientIP = "Client IP Address:{}".format(address)

            print(clientMsg)
            print(clientIP)

            # Sending a reply to client

            self.s.sendto(bytesToSend, address)


if __name__ == "__main__":
    # UNCOMMENT BELOW FOR SERVER TEST
    # server = SocketServer()
    # UNCOMMENT BELOW FOR CLIENT TEST
    # while True:
    #     time_start = time.time()
    #     client = SocketClient(host="bot")
    #     client.send_message("hello server")
    #     print(client.receive_message())
    #     time_end = time.time()
    #     print("fps: {}".format(1.0 / (time_end - time_start)))
