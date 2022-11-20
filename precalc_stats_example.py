#@title Calculate stats and save training set as binary .dat file in batch
#!/usr/bin/env python3
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import warnings
import numpy as np
import fid
import tqdm
# from scipy.misc import imread
import tensorflow as tf
from cv2 import imread,resize,INTER_CUBIC,cvtColor,COLOR_BGR2RGB
import pandas as pd
import numpy.lib as npl
from PIL import Image

# if os.path.isfile(os.path.join(data_path,'training_set_with_missing_images.dat')):
#   print("Part of compressed training set already exists. Removing... ")
#   os.remove(os.path.join(data_path,'training_set_with_missing_images.dat'))


########
# PATHS
########
data_path = '/Users/eden/Downloads/book dataset' # set path to training set images
# data_path="/content/drive/MyDrive/book dataset"
output_path = os.path.join(data_path,'fid_stats.npz') # path for where to store the statistics
compressed_path=os.path.join(data_path,'training_set_with_missing_images.dat')# path to store compressed training dataset
img_size=512

inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")
def compress_image():
    print("load images..", flush=True)
    df=pd.read_csv(os.path.join(data_path,"df_train.csv"))
    #preprocess every label in df_train into image path2
    concat_path=lambda x:os.path.join(data_path+'/images/images',str(x)+'.jpg')
    df=df[df.columns[0]].apply(concat_path)


    image_list=df.values.tolist()#df.values is numpy array
    images=None;failed_list=[]
    read=0;failed=0;compressed_length=0
    #initialize images 
    if os.path.isfile(compressed_path):
        try:
            images=np.fromfile(compressed_path,dtype=np.float32).reshape(-1,img_size,img_size,3)
        except:
            print('Error! Pre-compressed data shape doesn\'t match currently specified image size. Please delete the pre-compressed file. ')
            exit()
        if images.shape[0]<10000:#loading data over this threshold will cause OOE
            images=[images[i] for i in range(images.shape[0])]
            compressed_length=len(images)
            image_list=image_list[compressed_length:]
        else:
            images=[]
    else:
        images=[]
    print(f"Already compressed {compressed_length} images, continue writing into the file from that checkpoint....")


    #save in batch
    with open(compressed_path,mode='wb+') as f:
    for i in tqdm.tqdm(range(len(image_list))):
        #change to float32 from uint8(cv2.imread default) for compatiblity with TF model
        # print(image_list[i],os.path.isfile(image_list[i]))
        image=imread(image_list[i])
        if(image is not None):
            image=cvtColor(resize(image,(img_size,img_size),interpolation=INTER_CUBIC),COLOR_BGR2RGB).astype(np.float32)
            images.append(image)
            read+=1
            #save every 1000 iterations
            if i%1000==0 or i==len(image_list)-1:
                if i==0:
                    compressed_length+=1
                images=np.array(images)
                #write data into npy in batch
                print("Now saving: ",images.shape)
                images.tofile(f)
                print("  ||  Already saved: ", compressed_length+i," images")
                del images
                images=[]
        else:
            print("failed:",i)
            failed_list+=[image_list[i]]
            failed+=1

    images = np.array(np.fromfile(compressed_path).reshape(-1,img_size,img_size,3))
    print("%d images compressed" % len(images))

def calc_stats():
    try:
        images = np.array(np.fromfile(compressed_path).reshape(-1,img_size,img_size,3))
        print("%d images found and loaded"%len(images))
    except:
        print("Compressed data too large,OOE error! Exiting program...")
        exit()

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("Created!")

    print("calculte FID stats..", end=" ", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print("finished")
