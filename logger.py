
log_file = 'log.txt'

def write_to_log(log_file, who, message):
	with open(log_file, 'a+') as log:
		log.write(who + " : " + message + "\n")
