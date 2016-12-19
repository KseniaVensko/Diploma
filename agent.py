import socket

def initialize_sockets(port):
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	s.bind(('0.0.0.0', port))

	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST,1)
	return s, sock

port = 7777
listening_sock, sending_sock = initialize_sockets(port)

#while True:
#~ data = 'simplenetteaching,10,11,22,23,34,35,16,17,18,29,291,292,393,394,395'
#~ sending_sock.sendto(data, ('<broadcast>', port))
#~ mes = listening_sock.recv(1024)
#~ mes = listening_sock.recv(1024)
#~ if mes.startswith('simplenetsuccess'):
	#~ print mes
#~ else:
	#~ print 'another mes ' + mes
	# mes = generated:path:{path_to_im}:{name1}:{namen}
data = 'generate'
sending_sock.sendto(data, ('<broadcast>', port))
mes = listening_sock.recv(1024)
mes = listening_sock.recv(1024)
if mes.startswith('imagegenerated'):
	mes = mes.split(',')
	print mes[1]
else:
	print 'another mes ' + mes
