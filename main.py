from PIL import Image
import numpy as np
import CNN_wrapper
import random
import pickle
import os.path

############ Preprocessing ###################
imgs = []
labels = []
temp = []

def preprocess(file_name):
	count = 0
	with open(file_name,'r') as f:
		for line in f: 	
			line = line.split(' ')
			address = line[0]
			label = line[1]
			if label in temp:
				label = temp.index(label) 
			else:
				temp.append(label)
				label = temp.index(label)

			im = Image.open('flickr_logos_27_dataset_images/'+address)
			im = im.convert('L')
		
			pix = im.load()

			final_img = np.zeros((500,500,1))
			for i in range(im.size[0]):
				for j in range(im.size[1]):
					final_img[i,j] = [ pix[i,j]*1.0 ]
		
			imgs.append(final_img)				
			labels.append(label)
			count += 1
			if(qw%50==0):          #For checking on small dataset
				print "processed - ", count," items..."
				#break

		print
		print 'Preprocessing Completed.....'
		print

		return imgs, labels

############ Train #############################
file_name = 'train_data.dat'
if os.path.isfile(file_name) == False:
	imgs, labels = preprocess('flickr_logos_27_dataset_training_set_annotation.txt')
	imgs_and_labels = [imgs,labels]
	with open(file_name, "wb") as f:
		pickle.dump(imgs_and_labels, f)
else :
	with open(file_name,'rb') as f:
		imgs_and_labels = pickle.load(f)
	imgs, labels = imgs_and_labels
print 'Data Loaded'

imgs_vs_labels = zip(imgs,labels)
imgs_vs_labels = random.shuffle(imgs_vs_labels)
imgs, labels = zip(*imgs_vs_labels)[0], zip(*imgs_vs_labels)[1]

model = CNN_wrapper.CNN(inp_size=(500,500),n_classes=27)

model.train(train_set=(imgs,labels) )
	
############# Test ##############################

imgs, labels = preprocess('flickr_logos_27_dataset_query_set_annotation.txt')

print 'Accuracy: ',model.test(train)
