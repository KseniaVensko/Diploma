import socket
import socket_utils
from socket_utils import send_tcp_command, recv_tcp_command
from socket_utils import send_mes, recv_mes
import argparse
import os
import logger

generate_command = 'generate'
generate_success = 'imagegenerated'
recognize_command = 'recognize'
recognize_success = 'seenobjects'
generate_teach_command = 'generatingnetteaching'
recognize_teach_command = 'objectteaching'
generate_save_command = 'save_generate_model'
recognize_save_command = 'save_recognize_model'

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()

port = vars(options)['port']
my_name = "agent"
log_file = 'loggers/agent_logger.txt'

def accept_tcp_connections():
	generating_addr = recognition_addr = None
	i=0
	while i < 2:
		sock, addr = s.accept()
		data = sock.recv(1024)
		if "recognize_net" in data:
			print "recognize accepted"
			recognition_addr = addr
			recognition_s = sock
			i+=1
		elif "generate_net" in data:
			print "generate accepted"
			generating_addr = addr
			generating_s = sock
			i+=1
	return generating_addr, generating_s, recognition_addr, recognition_s
	
def accept_connections():
	generating_addr = recognition_addr = None
	i=0
	while i < 2:
		data, addr = s.recvfrom(1024)
		if "recognize_net" in data:
			print "recognize accepted"
			recognition_addr = addr
			i+=1
		elif "generate_net" in data:
			print "generate accepted"
			generating_addr = addr
			i+=1
	return generating_addr, recognition_addr
			
def send_command_new(data, addr):
	s.sendto(data, addr)
	data, addr = s.recvfrom(1024)
	return data

#s = socket_utils.initialize_server_socket_tcp('',port)
s = socket_utils.initialize_server_socket(port)

print("Listening...")

#generating_addr, generating_s, recognition_addr, recognition_s = accept_tcp_connections()
generating_addr, recognition_addr = accept_connections()

generating_miss = 0
recognizing_miss = 0
more_or_eq_than_half_collage_recornized_count = 0

for i in range(5):
	print 'sending generate command'

	#~ send_tcp_command(generate_command, generating_s)
	#~ mes = recv_tcp_command(generating_s)

	send_mes(s, generate_command, generating_addr)
	mes, addr = recv_mes(s)
	
	if mes.startswith(generate_success):
		print 'received answer'

		#send_tcp_command('waiting', generating_s)
		send_mes(s,'waiting', generating_addr)
		
		#name = socket_utils.receive_tcp_image(generating_s)
		name = socket_utils.receive_image(s)
		print name
		logger.write_to_log(log_file,my_name, "received generated image name " + name)
	else:
		print 'after generate received another mes ' + mes
		continue

	print 'sending recognize command'
	#~ send_tcp_command(recognize_command + ',' + name, recognition_s)
	#~ mes = recv_tcp_command(recognition_s)
	send_mes(s, recognize_command + ',' + name, recognition_addr)
	mes, addr = recv_mes(s)
	
	if 'waiting' in mes:
		#socket_utils.send_tcp_image(name, recognition_s)
		socket_utils.send_image(name, recognition_addr, s)
	else:
		print 'after recognize received another mes ' + mes
		continue
		
	#mes = recv_tcp_command(recognition_s)
	mes,addr = recv_mes(s)

	if mes.startswith(recognize_success):
		print 'received predictions'
		mes = mes.split(':')
		answer = mes[1].split(',')
		print answer
		objects = os.path.splitext(os.path.basename(name))[0].split("_")

		if set(objects) != set(answer):
			print "prediction was not correct"
			matches_percentage = float(len(set(objects) & set(answer))) / float(len(objects))

			if matches_percentage >= 0.5:
				more_or_eq_than_half_collage_recornized_count += 1
			
			recognizing_miss += matches_percentage
			
			logger.write_to_log(log_file,my_name, "correct objects " + str(objects) + "are not equal recognized objects " + str(answer))
			print "sending teaching commands"
			
			#~ send_tcp_command(generate_teach_command + ',' + str(True), generating_s)
			#~ mes = recv_tcp_command(generating_s)
			send_mes(s, generate_teach_command + ',' + str(True), generating_addr)
			mes,addr = recv_mes(s)
			
			print mes
			
			#~ send_tcp_command(recognize_teach_command + ',' + name + ',' + ','.join(objects), recognition_s)
			#~ mes = recv_tcp_command(recognition_s)
			send_mes(s, recognize_teach_command + ',' + name + ',' + ','.join(objects), recognition_addr)
			mes,addr = recv_mes(s)
			
			print mes
		else:
			generating_miss += 1

			print "sending generate teach command"
			
			#~ send_tcp_command(generate_teach_command + ',' + str(False), generating_s)
			#~ mes = recv_tcp_command(generating_s)
			send_mes(s, generate_teach_command + ',' + str(False), generating_addr)
			mes, addr = recv_tcp_command(s)
			
			
			print mes
	else:
		print 'instead of predictions receives another mes ' + mes

#mes = send_command_new(generate_save_command + ',' + 'generating_model.h5', generate_addr)
#mes = send_command_new(recognize_save_command + ',' + 'recognize_model.h5', recognize_addr)
print "recognition miss " + str(recognizing_miss)
print "generating miss " + str(generating_miss)
print "more_or_eq_than_half_collage_recornized_count " + str(more_or_eq_than_half_collage_recornized_count)
