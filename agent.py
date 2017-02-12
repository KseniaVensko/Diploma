import socket
import socket_utils
from socket_utils import send_tcp_command, recv_tcp_command
from socket_utils import send_mes, recv_mes
import argparse
import os
import logger
import timeit
import json
import time

generate_command = 'generate'
generate_success = 'imagegenerated'
recognize_command = 'recognize'
recognize_success = 'seenobjects'
generate_teach_command = 'generatingnetteaching'
recognize_teach_command = 'objectteaching'
generate_save_command = 'save_generate_model'
recognize_save_command = 'save_recognize_model'

script_path = os.path.dirname(os.path.abspath(__file__))
log_file = script_path + '/loggers/agent_logger.txt'

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7777)
parser.add_argument("--count", type=int, default=5)
parser.add_argument("--metrics_file", type=str, default=script_path + '/agent_metrics.json')
parser.add_argument("--min_recognition_time", type=float, default=0)
options = parser.parse_args()

port = vars(options)['port']
count = vars(options)['count']
metrics_file = vars(options)['metrics_file']
min_recognition_time = vars(options)['min_recognition_time']
my_name = "agent"

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

def write_metrics_to_json(metrics, file_name):
	print "writing metrics to "  + file_name
	with open(file_name, 'w') as f:
		json.dump(metrics, f)	

#s = socket_utils.initialize_server_socket_tcp('',port)
s = socket_utils.initialize_server_socket(port)

print("Listening...")

#generating_addr, generating_s, recognition_addr, recognition_s = accept_tcp_connections()
generating_addr, recognition_addr = accept_connections()

metrics = {}
metrics['Status'] = True
metrics['Runs'] = []

generating_miss = 0
recognizing_miss = 0
more_or_eq_than_half_collage_recornized_count = 0
recognition_time = 0

try:
	for i in range(count):
		one_iteration_metrics = {}
		print 'sending generate command'

		#~ send_tcp_command(generate_command, generating_s)
		#~ mes = recv_tcp_command(generating_s)
		
		start_time = timeit.default_timer()
		
		send_mes(s, generate_command, generating_addr)
		mes, addr = recv_mes(s)
		
		one_iteration_metrics['generation_time'] = timeit.default_timer() - start_time
		
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
		
		time_gap = timeit.default_timer() - recognition_time
		if time_gap < min_recognition_time:
			time.sleep(min_recognition_time - time_gap)
			
		print 'sending recognize command'
		#~ send_tcp_command(recognize_command + ',' + name, recognition_s)
		#~ mes = recv_tcp_command(recognition_s)
		
		send_mes(s, recognize_command + ',' + name, recognition_addr)
		mes, addr = recv_mes(s)
		
		if 'waiting' in mes:
			#socket_utils.send_tcp_image(name, recognition_s)
			socket_utils.send_image(name, recognition_addr, s)
			
			start_time = timeit.default_timer()
		else:
			print 'after recognize received another mes ' + mes
			continue
			
		#mes = recv_tcp_command(recognition_s)
		mes,addr = recv_mes(s)
		recognition_time = timeit.default_timer()
			
		one_iteration_metrics['recognition_time'] = timeit.default_timer() - start_time

		if mes.startswith(recognize_success):
			print 'received predictions'
			mes = mes.split(':')
			answer = filter(None, mes[1].split(','))
			print answer
			# filter removes Falsish elements ('', False, None, [], ...)
			objects = filter(None, os.path.splitext(os.path.basename(name))[0].split("_"))
			
			one_iteration_metrics['generated_objects'] = objects
			one_iteration_metrics['guessed_objects'] = answer
			one_iteration_metrics['incorrect_guessed_objects'] = list(set(answer) - set(objects))
			
			if set(objects) != set(answer):
				print "prediction was not correct"
				matches_percentage = float(len(set(objects) & set(answer))) / float(len(objects))
				one_iteration_metrics['guessing_percentage'] = matches_percentage

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
				mes, addr = recv_mes(s)
				
				
				print mes
		else:
			print 'instead of predictions receives another mes ' + mes
			continue
		
		metrics['Runs'].append(one_iteration_metrics)
except (KeyboardInterrupt, SystemExit):
	metrics['Status'] = False
	raise
except:
	metrics['Status'] = False
finally:
	write_metrics_to_json(metrics, metrics_file)

send_mes(s, generate_save_command + ',' + 'generating_model.h5', generating_addr)
mes, addr = recv_mes(s)
send_mes(s, recognize_save_command + ',' + 'recognize_model.h5', recognition_addr)
mes, addr = recv_mes(s)

print "recognition miss " + str(recognizing_miss)
print "generating miss " + str(generating_miss)
print "more_or_eq_than_half_collage_recornized_count " + str(more_or_eq_than_half_collage_recornized_count)
