from keras.models import load_model
import argparse
import socket_utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="recognition_model.h5")
parser.add_argument("--port", type=int, default=7777)
options = parser.parse_args()

model_file = vars(options)['model']
port = vars(options)['port']

listening_sock, sending_sock = socket_utils.initialize_sockets(port)
model = load_model(model_file)
