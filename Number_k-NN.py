import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from scipy.io import loadmat
import matplotlib.pyplot as plt
mnist = loadmat("mnist-original")
print(type(mnist))
'''Suite aux problemes avec fetch_openml de sklearn
 on uttilisera loadmat() de scipy'''

# Redimensionnement
mnist_data = mnist["data"].T
mnist_target = mnist["label"][0]

# On selectionne un echantillon au hasard
sample = np.random.randint(70000, size=5000)
data = mnist_data[sample]
target = mnist_target[sample]

# Separation train/test
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

# ici on va regarder l'erreur pour ckaque valeur de k et choisir le meilleur
'''
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)

error = 1 - knn.score(xtest, ytest)
print('Erreur: %f' % error)

errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
plt.plot(range(2,15), errors, 'o-')
plt.show()
'''

# On récupère le classifieur le plus performant
knn = neighbors.KNeighborsClassifier(4)
knn.fit(xtrain, ytrain)

# On récupère les prédictions sur les données test
predicted = knn.predict(xtest)

# On redimensionne les données sous forme d'images
images = xtest.reshape((-1, 28, 28))


# Si on veux on appercu des predictions :
'''
# On selectionne un echantillon de 12 images au hasard
select = np.random.randint(images.shape[0], size=12)

# On affiche les images avec la prédiction associée
fig,ax = plt.subplots(3,4)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: {}'.format( predicted[value]) )

plt.show()
'''


# on récupère les données mal prédites
misclass = (ytest != predicted)
misclass_images = images[misclass, :, :]
misclass_predicted = predicted[misclass]

# on sélectionne un échantillon de ces images
select = np.random.randint(misclass_images.shape[0], size=12)

# on affiche les images et les prédictions (erronées) associées à ces images
for index, value in enumerate(select):
    plt.subplot(3, 4, index + 1)
    plt.axis('off')
    plt.imshow(misclass_images[value], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title('Predicted: {}'.format(misclass_predicted[value]))

plt.show()
