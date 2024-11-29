# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report

# Étape 1 : Chargement et préparation des données
# Assurez-vous de remplacer "votre_fichier.csv" et les colonnes par les noms appropriés


features = ['colonne1', 'colonne2', 'colonne3']  # Colonnes de caractéristiques
target = 'colonne_cible'  # Colonne cible
X = data[features]
y = data[target]

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Étape 2 : Initialisation du modèle d'arbre de décision
model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

# Étape 3 : Entraînement du modèle
model.fit(X_train, y_train)

# Étape 4 : Création et entraînement d’un nouvel arbre de décision
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Étape 5 : Prédiction sur l’ensemble de test
y_pred = model.predict(X_test)

# Étape 6 : Évaluation des prédictions
print("Valeurs réelles :", y_test.to_numpy())
print("Valeurs prédites :", y_pred)

# Score de précision
score = model.score(X_test, y_test)
print("Précision :", score)

# Étape 7 : Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Rain', 'Sun', 'Fog', 'Drizzle', 'Snow'], 
            yticklabels=['Rain', 'Sun', 'Fog', 'Drizzle', 'Snow'])
plt.title('Matrice de Confusion')
plt.xlabel('Valeurs prédites')
plt.ylabel('Valeurs réelles')
plt.show()

# Étape 8 : Rapport de classification
report = classification_report(y_test, y_pred, target_names=['Rain', 'Sun', 'Fog', 'Drizzle', 'Snow'])
print(report)

# Étape 9 : Visualisation de l’arbre de décision
plt.figure(figsize=(50, 15), dpi=200)
plot_tree(model, 
          filled=True, 
          feature_names=X_train.columns, 
          class_names=['Rain', 'Sun', 'Fog', 'Drizzle', 'Snow'], 
          rounded=True, 
          fontsize=14)
plt.tight_layout()
plt.savefig("tree_model.png", dpi=200)
plt.show()

# Étape 10 : Recherche d’hyperparamètres optimaux avec GridSearchCV
param_grid_tree = {'max_depth': range(1, 10)}
grid_tree = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_tree, cv=5)
grid_tree.fit(X_train, y_train)
print("Meilleure profondeur maximale :", grid_tree.best_params_)

# Étape 11 : Utilisation de la profondeur optimale et visualisation
optimal_depth = grid_tree.best_params_['max_depth']
optimal_model = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
optimal_model.fit(X_train, y_train)

plt.figure(figsize=(50, 15), dpi=200)
plot_tree(optimal_model, 
          filled=True, 
          feature_names=X_train.columns, 
          class_names=['Rain', 'Sun', 'Fog', 'Drizzle', 'Snow'], 
          rounded=True, 
          fontsize=14)
plt.tight_layout()
plt.savefig("optimal_tree_model.png", dpi=200)
plt.show()
