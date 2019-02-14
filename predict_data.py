# this file essentially accepts a input from the user an image file, resizes it
# and tries to predict which character it is. It loads the trained model for this
# and computes the prediction on this model. 

import cv2
import sys
from keras.models import load_model
import numpy as np

# size of the image
img_width, img_height = 32,32
# labels for all classes
label = {0: '01_ka', 1: '02_kha', 2: '03_ga', 3: '04_gha', 4: '05_kna', 5: 'character_06_cha',
                    6: '07_chha', 7: '08_ja', 8: '09_jha', 9: '10_yna',
                    10: '11_taamatar',
                    11: '12_thaa', 12: '13_daa', 13: '14_dhaa', 14: '15_adna', 15: '16_tabala', 16: '17_tha',
                    17: '18_da',
                    18: '19_dha', 19: '20_na', 20: '21_pa', 21: '22_pha',
                    22: '23_ba',
                    23: '24_bha', 24: '25_ma', 25: '26_yaw', 26: '27_ra', 27: '28_la', 28: '29_waw', 29: '30_motosaw',
                    30: '31_petchiryakha',31: '32_patalosaw', 32: '33_ha',
                    33: '34_chhya', 34: '35_tra', 35: '36_gya', 36: 'd_0', 37: 'd_1', 38: 'd_2', 39: 'd_3', 40: 'd_4', 
                    41: 'd_5', 42: 'd_6', 43: 'd_7', 44: 'd_8', 45: 'd_9', 46: 'CHECK'}

# read the image given my the user and process it and reshaping it to size
img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (img_width,img_height))
img = np.array(img, dtype=np.float32)
img = np.reshape(img, (-3,img_width,img_height,3))
print ("processed: " + str(img.shape))
# load the trained model
model = load_model('my_test_model2.2.h5')
# use the predict function and print out to which class it belongs to.
pred_probab = model.predict(img)[0]
pred_class = list(pred_probab).index(max(pred_probab))
print ( max(pred_probab), pred_probab )
print(str(label[pred_class]))



# ------------------------------------ #
# using predict.generator

# from keras.preprocessing.image import ImageDataGenerator

# validation_dir = 'data/Validation'

# train_samples = 78200 # 1700x46
# validation_samples = 13754 # 300x40 + 294x6

# test_datagen = ImageDataGenerator(rescale=1./255)

# validation_generator = test_datagen.flow_from_directory(
# 	validation_dir,
# 	target_size=(img_width,img_height),
# 	batch_size=32,
# 	class_mode='categorical')

# model = load_model('my_test_model2.2.h5')

# predict = model.predict_generator(validation_generator, validation_samples)

# print predict