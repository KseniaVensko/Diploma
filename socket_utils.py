import socket
import os

# TODO: not obvious constant, maybe move it to func argument
buf = 512*512*3

def initialize_server_socket(port):
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	s.bind(('', port))
	
	return s
	
def initialize_client_socket(port):
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	
	return s

def send_mes_to_client(s, data):
	s.send(data)

def client_thread(conn):
    conn.send("Welcome to the Server. Type messages and press enter to send.\n")

    while True:
        data = conn.recv(1024)
        if not data:
            continue
        reply = "OK . . " + data
        print reply
        conn.sendall(reply)
    conn.close()
	
def initialize_server_socket_tcp(addr,port):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	s.bind((addr, port))
	s.listen(10)
	
	return s
	
def initialize_client_socket_tcp(addr,port):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	s.connect((addr, port))
	
	return s

def send_tcp_command(data, s):
	s.send(data)

def recv_tcp_command(s):
	s.recv(1024)
	return s

def send_image(file_name, addr, s):
	# send name
	s.sendto(file_name, addr)
	buf = 1024
	with open(file_name, 'rb') as f:
		data = f.read(buf)
		while data:
			s.sendto(data, addr)
			data = f.read(buf)
	s.sendto("end.", addr)

def receive_image(s):
	# receive name
	name,addr = s.recvfrom(1024)
	folder, file_name = os.path.split(name)
	if not os.path.exists(folder):
		os.makedirs(folder)
	buf = 1024
	im = ''
	data,addr = s.recvfrom(buf)
	while data != "end.":
		im += data
		data,addr = s.recvfrom(buf)
	with open(name.strip(), 'wb') as f:
		f.write(im)

	return name.strip()
	
def initialize_sockets(port):
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	s.bind(('0.0.0.0', port))

	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST,1)
	return s, sock
