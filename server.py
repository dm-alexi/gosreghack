import numpy as np
import skimage.io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,Dropout, Conv2D,Conv2DTranspose, BatchNormalization, Activation,AveragePooling2D,GlobalAveragePooling2D, Input, Concatenate, MaxPool2D, Add, UpSampling2D, LeakyReLU,ZeroPadding2D
from tensorflow.keras.models import Model
import cv2
import efficientnet.tfkeras as efn
from matplotlib import pyplot as plt
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):
    x_deep= Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)
    x_deep = BatchNormalization()(x_deep)   
    x_deep = LeakyReLU(alpha=0.1)(x_deep)
    x = Concatenate()([x_shallow, x_deep])
    x=Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)
    x = BatchNormalization()(x)   
    x = LeakyReLU(alpha=0.1)(x)
    return x

def cbr(x, out_layer, kernel, stride):
    x=Conv2D(out_layer, kernel_size=kernel, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def resblock(x_in,layer_n):
    x=cbr(x_in,layer_n,3,1)
    x=cbr(x,layer_n,3,1)
    x=Add()([x,x_in])
    return x

def create_model(input_shape, aggregation=True):
    input_layer = Input(input_shape)
    input_layer_1=AveragePooling2D(2)(input_layer)
    input_layer_2=AveragePooling2D(2)(input_layer_1)

    x_0= cbr(input_layer, 16, 3, 2)
    concat_1 = Concatenate()([x_0, input_layer_1])

    x_1= cbr(concat_1, 32, 3, 2)
    concat_2 = Concatenate()([x_1, input_layer_2])

    x_2= cbr(concat_2, 64, 3, 2)
    
    x=cbr(x_2,64,3,1)
    x=resblock(x,64)
    x=resblock(x,64)
    
    x_3= cbr(x, 128, 3, 2)
    x= cbr(x_3, 128, 3, 1)
    x=resblock(x,128)
    x=resblock(x,128)
    x=resblock(x,128)
    
    x_4= cbr(x, 256, 3, 2)
    x= cbr(x_4, 256, 3, 1)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
    x=resblock(x,256)
 
    x_5= cbr(x, 512, 3, 2)
    x= cbr(x_5, 512, 3, 1)
    
    x=resblock(x,512)
    x=resblock(x,512)
    x=resblock(x,512)
    
    x_1= cbr(x_1, output_layer_n, 1, 1)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_2= cbr(x_2, output_layer_n, 1, 1)
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)
    x_3= cbr(x_3, output_layer_n, 1, 1)
    x_3 = aggregation_block(x_3, x_4, output_layer_n, output_layer_n) 
    x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)
    x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

    x_4= cbr(x_4, output_layer_n, 1, 1)

    x=cbr(x, output_layer_n, 1, 1)
    x= UpSampling2D(size=(2, 2))(x)

    x = Concatenate()([x, x_4])
    x=cbr(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)

    x = Concatenate()([x, x_3])
    x=cbr(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)

    x = Concatenate()([x, x_2])
    x=cbr(x, output_layer_n, 3, 1)
    x= UpSampling2D(size=(2, 2))(x)
    x = Concatenate()([x, x_1])
    x=Conv2D(output_layer_n, kernel_size=3, strides=1, padding="same")(x)
    out = Activation("sigmoid")(x)
    
    model=Model(input_layer, out)
    return model

def NMS_all(predicts,category_n, pred_out_h, pred_out_w, score_thresh,iou_thresh):
    y_c=predicts[...,category_n]+np.arange(pred_out_h).reshape(-1,1)
    x_c=predicts[...,category_n+1]+np.arange(pred_out_w).reshape(1,-1)
    height=predicts[...,category_n+2]*pred_out_h
    width=predicts[...,category_n+3]*pred_out_w

    count=0
    for category in range(category_n):
        predict=predicts[...,category]
        mask=(predict>score_thresh)
        if mask.all==False:
            continue
        box_and_score=NMS(predict[mask],y_c[mask],x_c[mask],height[mask],width[mask],iou_thresh,pred_out_h, pred_out_w)
        box_and_score=np.insert(box_and_score,0,category,axis=1)
        if count==0:
            box_and_score_all=box_and_score
        else:
            box_and_score_all=np.concatenate((box_and_score_all,box_and_score),axis=0)
        count+=1
    score_sort=np.argsort(box_and_score_all[:,1])[::-1]
    box_and_score_all=box_and_score_all[score_sort]

    _,unique_idx=np.unique(box_and_score_all[:,2],return_index=True)
    
    return box_and_score_all[sorted(unique_idx)]
  
def NMS(score,y_c,x_c,height,width,iou_thresh,pred_out_h, pred_out_w,merge_mode=False):
    if merge_mode:
        score=score
        top=y_c
        left=x_c
        bottom=height
        right=width
    else:
        score=score.reshape(-1)
        y_c=y_c.reshape(-1)
        x_c=x_c.reshape(-1)
        height=height.reshape(-1)
        width=width.reshape(-1)
        size=height*width

        top=y_c-height/2
        left=x_c-width/2
        bottom=y_c+height/2
        right=x_c+width/2

        inside_pic=(top>0)*(left>0)*(bottom<pred_out_h)*(right<pred_out_w)
        outside_pic=len(inside_pic)-np.sum(inside_pic)
        
        normal_size=(size<(np.mean(size)*20))*(size>(np.mean(size)/20))
        score=score[inside_pic*normal_size]
        top=top[inside_pic*normal_size]
        left=left[inside_pic*normal_size]
        bottom=bottom[inside_pic*normal_size]
        right=right[inside_pic*normal_size]

    score_sort=np.argsort(score)[::-1]
    score=score[score_sort]  
    top=top[score_sort]
    left=left[score_sort]
    bottom=bottom[score_sort]
    right=right[score_sort]

    area=((bottom-top)*(right-left))

    boxes=np.concatenate((score.reshape(-1,1),top.reshape(-1,1),left.reshape(-1,1),bottom.reshape(-1,1),right.reshape(-1,1)),axis=1)

    box_idx=np.arange(len(top))
    alive_box=[]
    while len(box_idx)>0:
        alive_box.append(box_idx[0])

        y1=np.maximum(top[0],top)
        x1=np.maximum(left[0],left)
        y2=np.minimum(bottom[0],bottom)
        x2=np.minimum(right[0],right)

        cross_h=np.maximum(0,y2-y1)
        cross_w=np.maximum(0,x2-x1)
        still_alive=(((cross_h*cross_w)/area[0])<iou_thresh)
        if np.sum(still_alive)==len(box_idx):
            print("error")
            print(np.max((cross_h*cross_w)),area[0])
        top=top[still_alive]
        left=left[still_alive]
        bottom=bottom[still_alive]
        right=right[still_alive]
        area=area[still_alive]
        box_idx=box_idx[still_alive]
    return boxes[alive_box]

def visualize(box_and_score,img):
    boxes = []
    scores = []
    colors= [(0,0,255), (255,0,0), (0,255,255), (0,127,127), (127,255,127), (255,255,0)]
    classes = ["car", "motor", "person", "bus", "truck", "bike"]
    number_of_rect=np.minimum(500,1)

    for i in reversed(list(range(number_of_rect))):
        predicted_class, score, top, left, bottom, right = box_and_score[i,:]

        top = np.floor(top).astype('int32')
        left = np.floor(left).astype('int32')
        bottom = np.floor(bottom).astype('int32')
        right = np.floor(right).astype('int32')

        predicted_class = int(predicted_class)

        label = '{:.2f}'.format(score)
        cv2.rectangle(img, (left, top), (right, bottom), colors[predicted_class], 3)
        boxes.append([left, top, right-left, bottom-top])
        scores.append(score)
    
    return np.array(boxes), np.array(scores)

def analyze(model, carset, img_size):
	# car model detection
	image = skimage.io.imread('temp.jpg')
	image = image / 255.0
	image_ = cv2.resize(image, (img_size,img_size))
	image_ = np.reshape(image_,(1,img_size,img_size,3))
	car_prediction = model.predict(image, batch_size=1)
	car_prediction = np.argmax(car_prediction, axis = 1)[0]
	if car_prediction >= 0 and car_prediction < len(carset):
		result = carset[car_prediction]
	else:
		return ""
	# number coordinates detection
	pred_out_h=int(img_size/4)
	pred_out_w=int(img_size/4)
	category_n = 1
	predict = plate_coordinates_predictor.predict(image)
	predict = predict.reshape(pred_out_h,pred_out_w,(category_n+4))
	print_h, print_w = image.shape[1:3]
	box_and_score=NMS_all(predict,category_n, pred_out_h, pred_out_w, score_thresh=0.05,iou_thresh=0.05)
	box_and_score=box_and_score*[1,1,print_h/pred_out_h,print_w/pred_out_w,print_h/pred_out_h,print_w/pred_out_w]
	preds, scores = visualize(box_and_score,image_)
	#print ('plate coordinates: ', 'x_min =',preds[0][0], 'y_min =' , preds[0][1]+preds[0][3], 'x_max =', preds[0][0]+preds[0][2],'y_max =', preds[0][1])
	result['coord'] = [(preds[0][0], preds[0][1]+preds[0][3]), (preds[0][0]+preds[0][2], preds[0][1])]
	#plt.figure(figsize=(10,10))
	#plt.imshow(image_)
	return json.dump(result)

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
		self.wfile.write(analyze(model, carset, img_size).encode())

model = load_model('hakaton_b0_1.h5')
img_size = 512
category_n=1
output_layer_n=category_n+4
plate_coordinates_predictor = create_model(input_shape=(img_size,img_size,3))
plate_coordinates_predictor.load_weights('hakaton_plate_detection_best_checkpoint.h5')
carset = [{'brand' : 'KAMAZ', 'model' : '', 'veh_type' : 'C'},
		{'brand' : 'LADA', 'model' : 'PRIORA', 'veh_type' : 'B'},
		{'brand' : 'MAZDA', 'model' : '3', 'veh_type' : 'B'},
		{'brand' : 'RENAULT', 'model' : 'DUSTER', 'veh_type' : 'B'},
		{'brand' : 'SCANIA', 'model' : '', 'veh_type' : 'C'},
		{'brand' : 'TOYOTA', 'model' : 'RAV4', 'veh_type' : 'B'},
		{'brand' : 'VOLVO', 'model' : '', 'veh_type' : 'C'},
		{'brand' : 'VOLKSWAGEN', 'model' : 'TIGUAN', 'veh_type' : 'B'},
		{'brand' : 'VOLKSWAGEN', 'model' : 'POLO', 'veh_type' : 'B'},
		{'brand' : 'KIA', 'model' : 'RIO', 'veh_type' : 'B'},
		{'brand' : 'HYUNDAI', 'model' : 'SOLARIS', 'veh_type' : 'B'}]
"""
pred_out_h=int(img_size/4)
pred_out_w=int(img_size/4)
predict = plate_coordinates_predictor.predict(image)
predict = predict.reshape(pred_out_h,pred_out_w,(category_n+4))
print_h, print_w = image.shape[1:3]
box_and_score=NMS_all(predict,category_n, pred_out_h, pred_out_w, score_thresh=0.05,iou_thresh=0.05)
box_and_score=box_and_score*[1,1,print_h/pred_out_h,print_w/pred_out_w,print_h/pred_out_h,print_w/pred_out_w]
preds, scores = visualize(box_and_score,image_)
print ('plate coordinates: ', 'x_min =',preds[0][0], 'y_min =' , preds[0][1]+preds[0][3], 'x_max =', preds[0][0]+preds[0][2],'y_max =', preds[0][1])
plt.figure(figsize=(10,10))
plt.imshow(image_)
"""
server_address = ("", 8080)
httpd = HTTPServer(server_address, HttpProcessor)
httpd.serve_forever()


