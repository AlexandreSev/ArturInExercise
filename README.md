# ArturIn deep learning test

## Problème

Développer un modèle capable d'identifier le locuteur (Chirac vs. Mitterrand) d'un segment de discours politique.

Un dataset labellisé est fourni pour l'apprentissage

## Solution

Plusieurs modèles sont proposés pour répondre à ce problème:
  - Un naive bayes, pour baser la prédiction sur l'apparition de certains mots. Il peut être éxécuter grâce au script studyNB.py.
  - Un SVM linéaire sur un bag of word et un tf-ifd des phrases proposées. Il peut être éxécuter grâce au script studySVM.py.
  - Un classifieur sur le vecteur moyen d'une phrase. Il peut être éxécuter grâce au script studyMean.py.
  - Un CNN et un LSTM, qui peuvent être éxéctuer respectivement grâce à studyCNN.py et studyLSTM.py
  
## Résultats

Les résultats les plus probant sont ceux du naive bayes, qui se trouve dans le fichier résults.pkl, dans la racine de ce repo.

Sur le set de validation, nous avons trouvé un f1 score de:
 - 0.57 pour le naive bayes
 - 0.56 pour le SVM
 - 0.50 pour le CNN
 - 0.55 pour le LSTM
 
## Détails des solutions

### Naive Bayes et SVM

Le naive bayes est un modèle très simple comparant l'occurence de chaque mot dans les phrases de l'un et de l'autre. A partir de ces occurrences, il construit une probabilité qu'un mot, puis qu'une phrase appartienne à M. Chirac ou à M. Mitterand.

Le SVM linéaire (LinearSVC dans sklearn) est un modèle cherchant à determiner un hyperplan séparant les examples positifs des examples négatifs.

Ces modèles prennent en entrée un bag of word, c'est à dire une matrice dans laquelle le coefficient i,j représente le nombre de fois que le mot j apparait dans la phrase i. Ils peuvent aussi être utilisés avec un tf-idf de cette matrice, c'est à dire une renormalisation en fonction de la rareté d'un mot.

### Classifieur sur le vecteur moyen

Pour cette approche, un embedding pré-entrainé a été utilisé (cf http://fauconnier.github.io/). Il associe à chaque mot un vecteur de 200 dimensions. Nous avons moyénné les vecteurs des mots d'une phrase, et créé un classifieur XGBoost sur ce vecteur moyen.

Pour obtenir un résultat optimal, il aurait fallu essayé un grand nombre de paramètres, en jouant sur le nombre d'estimateur, la profondeur de ceux ci et la pénalisation de la perte, mais le temps nous a manqué pour effectuer un GridSearch aussi important.

### Réseaux Neuronaux

Pour les modèles de deep Learning, nous avons dû retravailler sur la tokenisation des mots. En effet, un grand nombre de mots présents dans les phrases n'étaient pas présent dans l'embedding. Nous avons donc créé un token "unknown" pour représenter les mots dont on avait pas l'embedding. Il a été initialisé avec la moyenne des vecteurs des autres mots. Nous avons aussi supprimé la ponctuation pour limiter le nombre de tokens "unknow".

Nous avons ensuite paddé les sequences pour qu'elles soient toutes de la même longueur et nous avons construit nos modèles sur ces séquences.

Les deux réseaux sont simples. Le réseau convolutionnel est un réseau à une couche de convolution composée de 512 filtres de tailles 5, puis d'une couche dense pour prédire le label. Le réseau récurrent est composée d'une couche LSTM de 128 neuronnes, puis d'une couche dense. 

On peut penser quand travaillant un peu plus sur la structure, nous aurions obtenus un score plus élevé avec l'un de ces deux modèles que avec le naive bayes.
