import socket

buf = 128*128*3

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

def send_image(file_name, addr):
	# send name
	s.sendto(file_name, addr)
	with open(file_name, 'rb') as f:
		data = f.read(buf)
	# send binary str
	s.sendto(data, addr)

def receive_image():
	# receive name
	name,addr = s.recvfrom(1024)

	data,addr = s.recvfrom(buf)
	with open(name.strip(), 'wb') as f:
		f.write(data)

	return name.strip()
	
def initialize_sockets(port):
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	s.bind(('0.0.0.0', port))

	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST,1)
	return s, sock
