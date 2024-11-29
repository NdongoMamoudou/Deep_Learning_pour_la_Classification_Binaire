# Prédiction des emails Spam 📧

Ce projet utilise le machine learning pour détecter si un email est un spam ou non, à partir de données textuelles. 

---

## 📝 Objectif

Créer un modèle capable de classifier les emails en deux catégories : 
- **Spam** : Email indésirable.
- **Ham** : Email légitime.

---

## 📊 Jeu de données

- **Source des données :** Ensemble de données contenant 193,850 emails étiquetés comme "Spam" ou "Ham".
- **Colonnes principales :**
  - `label` : Type d'email (Spam ou Ham).
  - `text` : Contenu textuel de l'email.

---

## 🔧 Étapes du projet

1. **Chargement des bibliothèques :**
   - Importation des bibliothèques nécessaires pour la manipulation des données, la visualisation et la création du modèle.

2. **Analyse exploratoire :**
   - Vérification des données manquantes.
   - Statistiques descriptives des colonnes.

3. **Prétraitement des données :**
   - Suppression des valeurs manquantes.
   - Transformation des labels textuels (Spam/Ham) en variables numériques.
   - Conversion des emails en séquences numériques à l'aide de `Tokenizer` et `pad_sequences`.

4. **Division du jeu de données :**
   - Division en ensembles d'entraînement et de test (80/20).

5. **Construction du modèle :**
   - Réseau de neurones séquentiel avec plusieurs couches denses (Keras).
   - Fonction d'activation : ReLU pour les couches cachées, Sigmoid pour la couche de sortie.

6. **Compilation et entraînement :**
   - Optimiseur : Adam.
   - Fonction de perte : Mean Squared Error.

7. **Évaluation du modèle :**
   - Prédictions sur l'ensemble de test.
   - Calcul des pertes et de la précision.

---

## 📈 Résultats

- **Précision du modèle :** Environ **52%** (Mean Squared Error).
- Le modèle nécessite probablement des améliorations (changement de la fonction de perte ou meilleure gestion des données d'entrée).

---

## 🚀 Améliorations possibles

1. Essayer une fonction de perte plus adaptée, comme `binary_crossentropy`.
2. Augmenter le nombre d'époques et ajuster le taux d'apprentissage.
3. Tester d'autres techniques de vectorisation de texte, comme TF-IDF ou Word2Vec.
4. Ajouter des techniques de régularisation (Dropout) pour éviter le surapprentissage.

---

## 🛠️ Technologies utilisées

- **Langage :** Python
- **Bibliothèques principales :**
  - **Manipulation des données :** Pandas, NumPy
  - **Visualisation :** Matplotlib, Seaborn
  - **Traitement du texte :** Keras Tokenizer, Sklearn CountVectorizer
  - **Machine Learning :** Keras, TensorFlow

---

## 📂 Structure des fichiers

- `spam_Emails_data.csv` : Jeu de données contenant les emails.
- `notebook_spam_detection.ipynb` : Script complet de l'analyse et du modèle.

---





___________________________________________________________________________________________________________________


🦠 Prédiction du Cancer
Ce projet utilise le Machine Learning pour prédire si un patient est atteint de cancer ou non, à partir de données médicales.

📝 Objectif
L'objectif de ce projet est de créer un modèle de machine learning capable de classer les cas médicaux en deux catégories :

Cancer : Le patient est atteint de cancer.
Non Cancer : Le patient n’est pas atteint de cancer.
📊 Jeu de données
Source des données : Dataset contenant des informations médicales sur les patients, telles que les caractéristiques cellulaires et histopathologiques, étiquetées comme "Cancer" ou "Non Cancer".
Colonnes principales :
label : Le type de cas (Cancer ou Non Cancer).
Diverses colonnes liées aux caractéristiques des cellules (par exemple, la texture, la zone, la concavité, etc.).
🔧 Étapes du projet
1) Chargement des bibliothèques :
Importation des bibliothèques nécessaires pour la manipulation des données, la création du modèle et l'évaluation des performances.




2) Analyse exploratoire des données :

• Vérification des données manquantes.
• Statistiques descriptives des colonnes.
• Visualisation des données pour observer les différences entre les classes "Cancer" et "Non Cancer".


3) Prétraitement des données :

• Traitement des labels : Transformation des labels "Cancer" et "Non Cancer" en variables numériques (1 et 0).
• Normalisation : Application d'un StandardScaler pour normaliser les données (afin de garantir que chaque caractéristique ait la même échelle).




4) Division du jeu de données :

    • Séparation des données en ensemble d'entraînement (80%) et ensemble de test (20%).

5) Construction du modèle :

    • Utilisation d'un classifieur Random Forest pour entraîner le modèle. Le Random Forest est un modèle robuste pour des problèmes de classification binaire.


6) Évaluation du modèle :

   • Prédictions sur l’ensemble de test et calcul de la précision, de la matrice de confusion et du rapport de classification.



📈 Résultats

Précision du modèle : Environ 96% (selon les résultats du modèle de Random Forest).
Le modèle peut être amélioré en testant d'autres algorithmes, ajustant les hyperparamètres ou en utilisant davantage de caractéristiques.
🚀 Améliorations possibles
Tester d'autres modèles comme les réseaux de neurones ou les SVM pour comparer les performances.
Ajuster les hyperparamètres du Random Forest pour améliorer les résultats (par exemple, le nombre d'arbres, la profondeur maximale, etc.).
Ajouter des techniques de régularisation ou de réduction de la dimensionnalité pour améliorer la généralisation du modèle.


🛠️ Technologies utilisées
Langage : Python
Bibliothèques principales :

Manipulation des données : Pandas, NumPy
Prétraitement : Scikit-learn (StandardScaler)
Machine Learning : Scikit-learn (RandomForestClassifier)
Visualisation : Matplotlib, Seaborn

📂 Structure des fichiers
cancer_data.csv : Jeu de données contenant les caractéristiques des patients et l'étiquette (Cancer ou Non Cancer).
notebook_cancer_detection.ipynb : Script complet pour l'analyse et la modélisation.



