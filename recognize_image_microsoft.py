import sys
import time 
import requests
import cv2
import operator
import numpy as np
import yaml

_url = 'https://westus.api.cognitive.microsoft.com/vision/v1.0/analyze'
_key = "c45f88d8f4204d65ac53265d898978fe" #Here you have to paste your primary key
_maxNumRetries = 10
_treshold = 0
_max_count = 10

def processRequest( json, data, headers, params ):

	"""
	Helper function to process the request to Project Oxford

	Parameters:
	json: Used when processing images from its URL. See API Documentation
	data: Used when processing image read from disk. See API Documentation
	headers: Used to pass the key information and the data type request
	"""

	retries = 0
	result = None

	while True:

		response = requests.request( 'post', _url, json = json, data = data, headers = headers, params = params )

		if response.status_code == 429: 

			print( "Message: %s" % ( response.json()['error']['message'] ) )

			if retries <= _maxNumRetries: 
				time.sleep(1) 
				retries += 1
				continue
			else: 
				print( 'Error: failed after retrying!' )
				break

		elif response.status_code == 200 or response.status_code == 201:

			if 'content-length' in response.headers and int(response.headers['content-length']) == 0: 
				result = None 
			elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str): 
				if 'application/json' in response.headers['content-type'].lower(): 
					result = response.json() if response.content else None 
				elif 'image' in response.headers['content-type'].lower(): 
					result = response.content
		else:
			print( "Error code: %d" % ( response.status_code ) )
			print( "Message: %s" % ( response.json()['error']['message'] ) )

		break
		
	return result

def read_image(path):
	with open( path, 'rb' ) as f:
		data = f.read()
	return data

def recognize_image(path):
	data = read_image(path)
	
	params = { 'visualFeatures' : 'Tags'} 
	
	headers = dict()
	headers['Ocp-Apim-Subscription-Key'] = _key
	headers['Content-Type'] = 'application/octet-stream'
	
	json = None
	
	result = processRequest(json, data, headers, params)
	tags = extract_tags(result["tags"])
	
	return tags

def extract_tags(tags):
	result = []
	
	# assume that tags are sorted by "confidence"
	for t in tags:
		if t["confidence"] > _treshold:
			result.append(t["name"])
			if len(result) >= _max_count:
				break
				
	return result
