import numpy as np
import skimage.io
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import efficientnet.tfkeras as efn
from http.server import HTTPServer, BaseHTTPRequestHandler

def predict(model, carset, img_size):
	image = skimage.io.imread('temp.jpg')
	image = image / 255.0
	image = cv2.resize(image, (img_size,img_size))
	image = np.reshape(image,(1,img_size,img_size,3))
	prediction = model.predict(image, batch_size=1)
	prediction = np.argmax(prediction, axis = 1)[0]
	if prediction >= 0 and prediction < len(carset):
		return(carset[prediction])

class HttpProcessor(BaseHTTPRequestHandler):
	def do_GET(self):
		self.send_response(200)
		self.end_headers()
		with open('index.html', "rb") as f:
			self.wfile.write(f.read())

	def do_POST(self):
		content_length = int(self.headers['Content-Length'])
		end_file = ('\r\n--' + self.headers['Content-Type'].partition("boundary=")[2]).encode()
		body = self.rfile.read(content_length).partition(b'\r\n\r\n')[2].partition(end_file)[0]
		self.send_response(200)
		self.end_headers()
		with open('temp.jpg', 'wb') as f:
			f.write(body)
		self.wfile.write(predict(model, carset, img_size).encode())

model = load_model('hakaton_b0.h5')
carset = ['KAMAZ_ALLKAMAZ_C', 'LADA_PRIORA_B', 'MAZDA_3_B', 'RENAULT_DUSTER_B', 'SCANIA_ALLSCANIA_C', 'TOYOTA_RAV4_B', 'VOLVO_ALLVOLVO_C', 'VOLKSWAGEN_TIGUAN_B', 'VOLKSWAGEN_POLO_B', 'KIA_RIO_B', 'HYUNDAI_SOLARIS_B']
img_size = 512

server_address = ("", 8080)
httpd = HTTPServer(server_address, HttpProcessor)
httpd.serve_forever()
