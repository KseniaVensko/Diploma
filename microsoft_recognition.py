import argparse
import socket_utils
from socket_utils import send_mes, recv_mes
import logger
import time
from recognize_image_microsoft import recognize_image
import os

my_name = "pretrained_recognition"
current_dir = os.path.dirname(os.path.abspath(__file__))
log_file = current_dir + '/loggers/recognition_logger.txt'
coef = 0.5
objects_count = 3
objects = {}
img_width, img_height = 128, 128

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7777)
parser.add_argument("--addr", type=str, default='127.0.0.1')
parser.add_argument("--key", type=str, default='', help='api key for microsoft computer vision service')
options = parser.parse_args()
addr = vars(options)['addr']
port = vars(options)['port']
key = vars(options)['key']
global s
# keys is array of objects i.e [bear,crocodile,...]
global keys

def initialize():
	logger.write_to_log(log_file,my_name, "load sockets and model")
	global s
	s = socket_utils.initialize_client_socket(port)
	send_mes(s, "recognize_net", (addr, port))
	global keys
	keys = []

	with open(current_dir + '/objects.txt') as f:
			for line in f:
				keys.append(line.rstrip('\n'))
	
	logger.write_to_log(log_file,my_name, "object keys " + str(keys))
	logger.write_to_log(log_file,my_name, "initialization complete")
	print "initialization complete"

def cast_tag_to_known_tags(tag):
	known_tags = {}
	known_tags['bear'] = 'bear'
	known_tags['crocodilian reptile'] = 'crocodile'
	known_tags['crocodile'] = 'crocodile'
	known_tags['dog'] = 'dog'
	known_tags['canids'] = 'dog'
	known_tags['horse'] = 'horse'
	known_tags['kangaroo'] = 'kangoro'
	known_tags['roo'] = 'kangoro'
	known_tags['lagomorph'] = 'rabbit'
	known_tags['rabbit'] = 'rabbit'
	known_tags['hare'] = 'rabbit'
	known_tags['sheep'] = 'sheep'

	if known_tags.has_key(tag):
		tag = known_tags[tag]
		
	return tag

def filter_tags(tags):
	print tags
	casted_tags = []
	for t in tags:
		casted_tags.append(cast_tag_to_known_tags(t))
	print "casted tags " + str(casted_tags)
	allowed_tags = ['bear', 'dog', 'crocodile', 'kangoro', 'horse', 'sheep', 'rabbit']
	result = []
	for t in casted_tags:
		if t in allowed_tags:
			result.append(t)
	print "result " + str(result)
	return result
	
initialize()

teach_command = 'objectteaching'
teach_success = 'recognitionsuccess'
recognize_command = 'recognize'
recognize_sucess = 'seenobjects'
recognize_save_command = 'save_recognize_model'

while True:
	mes, addr = recv_mes(s)
	
	logger.write_to_log(log_file,my_name, "received mes " + mes)
	
	if mes.startswith(teach_command):
		print "received teaching command"
		print "sending teaching success"
		send_mes(s, teach_success, addr)

	elif mes.startswith(recognize_command):
		print "received recognize command"
		send_mes(s, 'waiting', addr)
		path = socket_utils.receive_image(s)
		print path
		recognition_result = filter_tags(recognize_image(path, key))
		print "sending recognize success with objects " + ", ".join(recognition_result)
		data = recognize_sucess + ':' + ",".join(recognition_result)
		send_mes(s, data,addr)
		
	elif mes.startswith(recognize_save_command):
		mes = mes.split(',')
		path = mes[1]
		print "saving model to " + path
		send_mes(s, recognize_sucess, addr)

