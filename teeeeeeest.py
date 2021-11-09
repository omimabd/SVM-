from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

# nous avons deux categories dans notre base de donnes
Categories = ['voiture', 'bateau']
input_data = []  # tableau d'entrée
output_data = []  # tableau de sortie
database = 'DB2C'

for i in Categories:

    path = os.path.join(database, i)
    for img in os.listdir(path):
        image = imread(os.path.join(path, img))
        image = resize(image, (120, 120, 3))
        # pour applatir la matrice des images
        input_data.append(image.flatten())
        output_data.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
input_data = np.array(input_data)
output_data = np.array(output_data)
# dataframe pour structurer les données à 2 dimensions
df = pd.DataFrame(input_data)
df['output'] = output_data
x = df.iloc[:, :-1]  # return : le vecteur ligne jusqu'à la dernière colonne
# return : le vecteur ligne des valeurs de la dernière colonne
y = df.iloc[:, -1]

model = svm.SVC(kernel='linear', probability=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=77, stratify=y)
print('Splitted Successfully')
model.fit(x_train, y_train)
print('The Model is trained well with the given images')
y_pred = model.predict(x_test)
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
# --------------------
url = input('Enter URL of Image :')
img = imread(url)
plt.imshow(img)
plt.show()
img = resize(img, (120, 120, 3))
l = [img.flatten()]
probability = model.predict_proba(l)
for i, val in enumerate(Categories):
    print(f'{val} = {probability[0][i]*100}%')
print("The predicted image is : "+Categories[model.predict(l)[0]])
