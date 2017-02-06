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
generate_save_command = 'save_generate_model'
recognize_save_command = 'save_recognize_model'

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

def send_command(data):
	sending_sock.sendto(data, ('<broadcast>', port))
	mes = listening_sock.recv(1024)
	mes = listening_sock.recv(1024)
	return mes

generate_addr, recognize_addr = accept_connections()

generating_miss = 0
recognizing_miss = 0
more_or_eq_than_half_collage_recornized_count = 0
j = 0
for i in range(5):
	print 'sending generate command'
	mes = send_command_new(generate_command, generate_addr)
	if mes.startswith(generate_success):
		print 'received answer'
		s.sendto('waiting', generate_addr)
		name = socket_utils.receive_image(s)
		print name
		logger.write_to_log(log_file,my_name, "received generated image name " + name)
	else:
		print 'after generate received another mes ' + mes
		continue

	print 'sending recognize command'
	mes = send_command_new(recognize_command + ',' + name, recognize_addr)
	if 'waiting' in mes:
		socket_utils.send_image(name, recognize_addr, s)
	else:
		print 'after recognize received another mes ' + mes
		continue
	mes,addr = s.recvfrom(1024)
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
			mes = send_command_new(generate_teach_command + ',' + str(True), generate_addr)
			print mes
			mes = send_command_new(recognize_teach_command + ',' + name + ',' + ','.join(objects), recognize_addr)
			print mes
		else:
			generating_miss += 1

			print "sending generate teach command"
			mes = send_command_new(generate_teach_command + ',' + str(False), generate_addr)
			print mes
	else:
		print 'instead of predictions receives another mes ' + mes

mes = send_command_new(generate_save_command + ',' + 'generating_model.h5', generate_addr)
mes = send_command_new(recognize_save_command + ',' + 'recognize_model.h5', recognize_addr)
print "recognition miss " + str(recognizing_miss)
print "generating miss " + str(generating_miss)
print "more_or_eq_than_half_collage_recornized_count " + str(more_or_eq_than_half_collage_recornized_count)
