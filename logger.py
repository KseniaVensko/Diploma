
log_file = 'log.txt'

def write_to_log(who, message):
	with open(log_file, 'a+') as log:
		log.write(who + " : " + message + "\n")
