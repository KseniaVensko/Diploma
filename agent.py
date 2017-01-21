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
#listening_sock, sending_sock = socket_utils.initialize_sockets(port)
s = socket_utils.initialize_server_socket(port)

def accept_connections():
	generating_addr = recognition_addr = None
	i=0
	while i < 2:
		data, addr = s.recvfrom(1024)
		if "recognize" in data:
			recognition_addr = addr
			i+=1
		elif "generate" in data:
			generating_addr = addr
			i+=1
	return generating_addr, recognition_addr
			
def send_command_new(data, addr):
	s.sendto(data, addr)
	data, addr = s.recvfrom(1024)
	return data

def send_command(data):
	sending_sock.sendto(data, ('<broadcast>', port))
	mes = listening_sock.recv(1024)
	mes = listening_sock.recv(1024)
	return mes

generate_addr, recognize_addr = accept_connections()
#while True:
for i in range(4):
	mes = send_command_new(generate_command, generate_addr)
	if mes.startswith(generate_success):
		mes = mes.split(',')
		print mes[1]
		name = mes[1]
		
		logger.write_to_log(log_file,my_name, "received generated image name " + name)
	else:
		print 'after generate another mes ' + mes
		continue

	mes = send_command_new(recognize_command + ',' + name, recognize_addr)
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
			mes = send_command_new(generate_teach_command + ',' + str(True), generate_addr)
			mes = send_command(recognize_teach_command + ',' + name + ',' + ','.join(objects), recognize_addr)
			print "prediction was not correct"
		else:
			mes = send_command(generate_teach_command + ',' + str(False), generate_addr)
	else:
		print 'after recognize another mes ' + mes
