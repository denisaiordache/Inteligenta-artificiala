from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from pandas import read_csv
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import numpy as np
import plot as plt

#ETAPA 1 : preprocesarea datelor
#extragem informatiile (imaginile, respectiv label-urile) din fisierele de antrenare, validare si testare, apoi le stocam in numpy-arrays
images = []

#loading train data
trainFile = read_csv('train.txt')


for i in tqdm(range(trainFile.shape[0])):  #numarul de imagini
    img = image.load_img('train/' + trainFile['id'][i],target_size=(50, 50, 1), color_mode='grayscale')  #incarcam imaginea cu inaltimea si latimea spcificate in target size, al 3-lea parametru reprezentand num_channels, automat 1 cand folosim grayscale (3 la rgb de ex)
    #img.show()
    norm_img = image.img_to_array(img)/255  #normalizam imaginea
    images.append(norm_img)
    #plt.imshow(norm_img)

#convertim imaginile intr-un numpy array (mai eficienti, avem operatii predefinite pe ei)
train_data = np.array(images)
train_labels = to_categorical(trainFile['label'].values) #trebuie convertita la acest tip deoarece folosesc cce
#functia returneaza o matrice de len(vector) X numar clase
#print(train_labels)


#urmam aceiasi pasi pentru datele de validare, respectiv cele de testare
images = []

#loading validation

validationFile = read_csv('validation.txt')

for i in tqdm(range(validationFile.shape[0])):  #numarul de imagini
    img = image.load_img('validation/' + validationFile['id'][i],target_size=(50, 50, 1), color_mode='grayscale')
    #img.show()
    norm_img = image.img_to_array(img)/255  #normalizam imaginea
    images.append(norm_img)
    #plt.imshow(norm_img)

validation_data = np.array(images)
validation_labels = to_categorical(validationFile['label'].values)
#print(validation_labels)

images = []
testFile = read_csv('test.txt')
for i in tqdm(range(testFile.shape[0])):  #numarul de imagini
    img = image.load_img('test/' + testFile['id'][i],target_size=(50, 50, 1), color_mode='grayscale')
    #img.show()
    norm_img = image.img_to_array(img)/255  #normalizam imaginea
    images.append(norm_img)
    #plt.imshow(norm_img)

test_data = np.array(images)

#Etapa 2: crearea modelului
#model 1 
model = Sequential()  #strat de layere: single-input, single-output
model.add(Conv2D(16,(3, 3), input_shape=(50,50,1)))
# params: output space, dimensiunea ferestrei de convolutie, shape-ul imaginii
# feature map -> dupa ce este creata, valorile ce reprezinta imaginile sunt pasate unui layer de activare
#functia/layerul de activare ia valorile ce repr imaginea intr-o forma liniara - lista de numere (datorita layer-ului de convolutie)

model.add(Activation('relu'))
#relu = Rectified Linear unit
# cu param default returneaza max(x,0) -deviatia standard ReLU

model.add(MaxPooling2D(pool_size=(2, 2)))
#dupa ce datele sunt "activate", sunt trimise printr-un layer de Pooling
#preia informatia ce reprezinta imaginea, o micsoreaza, apoi abstractizeaza partile care i se par "irelevante"


model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))

#for final layers: densely connected layers:
#informatia trebuie compresata (flatten) intr-un vector
model.add(Flatten())

#straturile finale
model.add(Dense(128)) #param = nr de neuroni
model.add(Activation('relu'))
#A densely connected layer provides learning features from all the combinations of the features
# of the previous layer,
#whereas a convolutional layer relies on consistent features with a small repetitive field.(docum oficiala)

model.add(Dropout(0.5))
#layerul de dropout "renunta" la unitati din tensor, pentru a evita overfiyiing-ul
model.add(Dense(3))
#la final, parametrul layerului Dense trebuie sa fie egal cu nr claselor

model.add(Activation('softmax'))
#mapeaza outputul in [0,1], fiecare output da suma 1
#adesea aceasta functie de activare este folosita ca si layer final (impreuna cu categorical_corss)
#multi-classification in logistic regression model

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=2, validation_data=(validation_data, validation_labels))

test_labels = model.predict_classes(test_data)
#print(len(test_predictions))

validation_predictions = model.predict_classes(validation_data)
rounded_labels=np.argmax(validation_labels, axis=1)
#print(validation_predictions)
#print(rounded_labels)
cm = confusion_matrix(rounded_labels,validation_predictions)

print(cm)

#copiez continutul fisierului sample_submission, atribui predictiile coloanei cu header-ul 'label',apoi il convertesc la tipul csv
outputFile = read_csv('sample_submission.txt')
outputFile['label'] = test_labels
outputFile.to_csv('submission4.csv', header=True, index=False)

