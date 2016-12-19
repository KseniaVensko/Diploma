import socket
import socket_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()

port = vars(options)['port']

listening_sock, sending_sock = socket_utils.initialize_sockets(port)

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
