import socket
import socket_utils
import argparse
import os
import logger

generate_command = 'generate'
generate_success = 'imagegenerated'
recognize_command = 'recognize'
recognize_success = 'seenobjects'
generate_teach_command = 'generatingnetteaching'
recognize_teach_command = 'objectteaching'

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()

port = vars(options)['port']
my_name = "agent"
log_file = 'loggers/agent_logger.txt'
listening_sock, sending_sock = socket_utils.initialize_sockets(port)

def send_command(data):
	sending_sock.sendto(data, ('<broadcast>', port))
	mes = listening_sock.recv(1024)
	mes = listening_sock.recv(1024)
	return mes

#while True:
for i in range(4):
	mes = send_command(generate_command)
	if mes.startswith(generate_success):
		mes = mes.split(',')
		print mes[1]
		name = mes[1]
		
		logger.write_to_log(log_file,my_name, "received generated image name " + name)
	else:
		print 'after generate another mes ' + mes
		continue

	mes = send_command(recognize_command + ',' + name)
	if mes.startswith(recognize_success):
		mes = mes.split(':')
		print mes[1]
		print name
		objects = os.path.splitext(os.path.basename(name))[0].split("_")
		print 'lala'
		if mes[1] == []:
			print "prediction was not correct"
		elif set(objects) != set(mes[1]):
			logger.write_to_log(log_file,my_name, "correct objects " + str(objects) + "are not equal recognized objects " + str(mes[1]))
			mes = send_command(generate_teach_command + ',' + str(True))
			mes = send_command(recognize_teach_command + ',' + name + ',' + ','.join(objects))
			print "prediction was not correct"
		else:
			mes = send_command(generate_teach_command + ',' + str(False))
	else:
		print 'after recognize another mes ' + mes
