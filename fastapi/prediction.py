from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import utils_eff
from efficient import EfficientNet
from resnet import ResNet18

import torch
import torchvision.transforms as transforms


model_path = './last_model.pth'




model = None


def load_model():
	#model = tf.keras.applications.MobileNetV2(weights="imagenet")
	device = torch.device('cpu')
	blocks_args, global_params = utils_eff.efficientnet( num_classes=200)
	model = EfficientNet(blocks_args, global_params).to(device)
	#model = ResNet18(num_classes=200)
	state_dict = torch.load(model_path, map_location=device)
	model.load_state_dict(state_dict, strict=False)
	print("Model loaded")
	return model


def predict(image: Image.Image):
	global model
	if model is None:
		model = load_model()

	# load the image
	#image = Image.open(image)
	# summarize some details about the image
	data = np.asarray(image)
	img_resized = image.resize((64,64))
	    
	tfms = transforms.Compose([transforms.Resize((64,64)), transforms.CenterCrop((64,64)), 
	                           transforms.ToTensor(),
	                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
	img = tfms(image).unsqueeze(0)


	with torch.no_grad():
		logits = model(img)
		preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()
		print('-----')
	
	with open('./data/de_words.txt') as f:
		lines = f.readlines()
	with open('./data/wnids.txt') as f:
		codes = f.readlines()

	labels = []
	probs = []

	for idx in preds:
		label = lines[idx]
		hola = codes.index(codes[idx])
		label2 = lines[hola]
		print(label, label2, codes[idx], hola)
		prob = torch.softmax(logits, dim=1)[0, idx].item()
		print('{:<75} ({:.2f}%)'.format(label, prob*100))
		labels.append(label)
		probs.append(prob*100)
	#output = model(img)
	#pred = output.argmax(dim=1, keepdim=True) 
	#with open('./data/de_words.txt') as f:
		#lines = f.readlines()
	#print(lines[pred])
	#print(pred, output[pred], lines[pred])




	fig = plt.figure(figsize = (10, 5))
	 
	for i in range(len(labels)):
		labels[i] = labels[i][9:]
	# creating the bar plot
	plt.bar(labels, probs, color ='maroon',
	        width = 0.4)
	 
	plt.xlabel("Labels")
	plt.ylabel("Probability")
	plt.title("Output")
	#plt.show()

	return labels, probs, plt



def predict_class(image: Image.Image):
	global model
	if model is None:
		model = load_model()

	# load the image
	#image = Image.open(image)
	# summarize some details about the image
	data = np.asarray(image)
	img_resized = image.resize((64,64))
	    
	tfms = transforms.Compose([transforms.Resize((64,64)), transforms.CenterCrop((64,64)), 
	                           transforms.ToTensor(),
	                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
	img = tfms(image).unsqueeze(0)


	with open('./data/de_words.txt') as f:
		lines = f.readlines()

	output = model(img)
	pred = output.argmax(dim=1, keepdim=True) 
	with open('./data/de_words.txt') as f:
		lines = f.readlines()
	print(lines[pred])
	#print(pred, output[pred], lines[pred])




	return lines[pred]


def read_imagefile(file) -> Image.Image:
	image = Image.open(BytesIO(file))
	return image