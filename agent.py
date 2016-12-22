import socket
import socket_utils
import argparse
import os
import logger

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()

port = vars(options)['port']
my_name = "agent"
listening_sock, sending_sock = socket_utils.initialize_sockets(port)

#while True:
data = 'generate'
sending_sock.sendto(data, ('<broadcast>', port))
mes = listening_sock.recv(1024)
mes = listening_sock.recv(1024)
if mes.startswith('imagegenerated'):
	mes = mes.split(',')
	print mes[1]
	name = mes[1]
	logger.write_to_log(my_name, "received generated image name " + name)
else:
	print 'another mes ' + mes

logger.write_to_log(my_name, "asking to recognize image" + name)
data = 'recognize,' + name
sending_sock.sendto(data, ('<broadcast>', port))
mes = listening_sock.recv(1024)
mes = listening_sock.recv(1024)
if mes.startswith('seenobjects'):
	mes = mes.split(':')
	print mes[1]
	objects = os.path.splitext(name)[0].split("_")
	if (set(objects) != set(mes[1])):
		logger.write_to_log(my_name, "correct objects " + str(objects) + "are not equal recognized objects" + str(mes[1]))
		#~ data = 'simplenetteaching,10,11,22,23,34,35,16,17,18,29,291,292,393,394,395'
		#~ sending_sock.sendto(data, ('<broadcast>', port))
		#~ mes = listening_sock.recv(1024)
		#~ mes = listening_sock.recv(1024)
		#~ if mes.startswith('simplenetsuccess'):
			#~ print mes
		#~ else:
			#~ print 'another mes ' + mes
		print "did not guess"
else:
	print 'another mes ' + mes
