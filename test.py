import cv2
import numpy as np
import glob
import math

def norm(array):
    return array / np.linalg.norm(array)
#Fetch names of images
img_names=glob.glob("*.png")
img_names.extend(glob.glob("*.jpg"))
img_names.extend(glob.glob("*.jpeg"))
#training
#using haar cascade classifier for face detection
fc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
crop_img=[]
g=0
read_names=[]
for imgs in img_names:
    img_mat=cv2.imread(imgs, 0)
    faces = fc.detectMultiScale(img_mat, 1.3, 5)  #detect faces
    if(type(faces)!=tuple):
        read_names.append(imgs)
        cropped=img_mat[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]  #crop faces
        crop_img.append(cv2.resize(cropped,(128,128)))    #resizing the croped face
    g+=1
#convert image matrices to vector
img_vect=[]
y=0
for img in crop_img:
    img_vect.append(img.flatten())
    y+=1
img_vect=np.array(img_vect)
    
#finding mean of all vectors
mean_vect=np.mean(img_vect, axis = 0)
'''
for x in range(len(img_vect)):
        img_vect[x]=np.subtract(img_vect[x],mean_vect)
'''
'''
m=0
for img in img_vect:
    h=np.reshape(img,(128,128))
    cv2.imshow("a"+str(m), h)
    m+=1
'''
#covariance matrix
C=np.cov(img_vect)
#eigen vectors, values
ev,v=np.linalg.eig(C)
sorted_val=np.sort(ev)         #sortEigen values
sorted_vec=v[:,ev.argsort()]

#normalize eigen vectors
for x in range(len(sorted_vec)):
    sorted_vec[x]=norm(sorted_vec[x])

#take 18 best eigen vectors and project on higher dimension space
eigmat=[]
for al in range(18):
    eigmat.append(list(np.matmul(np.array(img_vect).transpose(),sorted_vec[al])))
eigmat=np.array(eigmat)
'''
y=0
for al in eigmat:
    h=np.reshape(al,(128,128))
    cv2.imshow("a"+str(y), h.astype('uint8'))
    y+=1
'''
#again normalize eigen vectors
for x in range(len(eigmat)):
    eigmat[x]=norm(eigmat[x])

#finding weights of each training image
tr_weights=[]
for img in img_vect:
    tr_weights.append([])
    for vect in eigmat:
        tr_weights[-1].append(img.dot(vect))


#testing
#Fetch names of  test images
test_img=glob.glob("tes/*.png")
test_img.extend(glob.glob("tes/*.jpg"))
test_img.extend(glob.glob("tes/*.jpeg"))
#using haar cascade classifier for face detection
tst_mat=[]
crop_img=[]
t_read_nms=[]
fc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for imgs in test_img:
    img_mat=cv2.imread(imgs, 0)
    faces = fc.detectMultiScale(img_mat, 1.3, 5)  #detect faces
    if(type(faces)!=tuple):
        t_read_nms.append(imgs)
        cropped=img_mat[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]  #crop faces
        crop_img.append(cv2.resize(cropped,(128,128)))    #resizing the croped face
    g+=1
#convert image matrices to vector
tst_vect=[]
for img in crop_img:
    tst_vect.append(img.flatten())

#mean vector
#for x in range(len(tst_vect)):
#        tst_vect[x]=np.subtract(tst_vect[x],mean_vect)
#finding weights


#finding weights of each test image
ts_weights=[]
for img in tst_vect:
    ts_weights.append([])
    for vect in eigmat:
        ts_weights[-1].append(img.dot(vect))

#comparing the weights of each test image to all training images
#the trining set image with least distance from test set image is theanswer for that test set image
for each in range(len(ts_weights)):
    dist_mat=[]
    min_val=100000000000
    indx=0
    for al in range(len(tr_weights)):
        dist=np.linalg.norm(np.array(ts_weights[each])-np.array(tr_weights[al]))
        dist_mat.append(dist)
        if(dist<min_val):
            min_val=dist
            indx=al
    print(t_read_nms[each])
    print(read_names[indx])        
