import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree#, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# import graphviz
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  
# metadata 
#print(adult.metadata) 
  
# variable information 
#print(adult.variables) 

print("Podgląd danych wejściowych:")
print(X.head())
print("\nPodgląd zmiennej celu:")
print(y.head())

X = X.fillna(X.mode().iloc[0])
y = y['income']

X_encoded = pd.get_dummies(X, drop_first=True)
y_encoded = y.replace({'<=50K': 0, '>50K': 1})
y_encoded = y_encoded.replace({'<=50K.': 0, '>50K.': 1}).infer_objects(copy=False)
y_encoded = y_encoded.astype(int)



X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nDokładność modelu na zbiorze testowym:", accuracy_score(y_test, y_pred))
print("\nRaport klasyfikacji")
print(classification_report(y_test, y_pred))

used_features = X_encoded.columns

feature_mapping = {
    "age": "Wiek",
    "workclass_Private": "Rodzaj pracy: sektor prywatny",
    "workclass_Self-emp-not-inc": "Rodzaj pracy: samozatrudnienie bez inkorporacji",
    "fnlwgt": "Waga finalna",
    "education-num": "Liczba lat edukacji",
    "education_Masters": "Edukacja: magister",
    "marital-status_Married-civ-spouse": "Stan cywilny: małżeństwo",
    "occupation_Exec-managerial": "Zawód: kierownictwo wykonawcze",
    "occupation_Farming-fishing": "Zawód: rolnictwo i rybołówstwo",
    "relationship_Husband": "Relacja: mąż",
    "race_White": "Rasa: biała",
    "sex_Male": "Płeć: mężczyzna",
    "capital-gain": "Zysk kapitałowy",
    "capital-loss": "Strata kapitałowa",
    "hours-per-week": "Godziny pracy tygodniowo",
    "native-country_United-States": "Kraj pochodzenia: Stany Zjednoczone",
    "native-country_South": "Kraj pochodzenia: południe"
}



cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns. heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.title("Macierz pomyłek")
plt.xlabel("Predykcja")
plt.ylabel("Rzeczywistość")
plt.show()

feature_importances = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)
feature_importances['Feature'] = feature_importances['Feature'].replace(feature_mapping)

print("\nZnaczenie cech:")
print(feature_importances)

non_zero_features = feature_importances[feature_importances['Importance'] > 0]
top_features = feature_importances.sort_values(by='Importance', ascending=False).head(10)

print("Cechy z niezerowym wpływem:")
print(non_zero_features)

feature_importances['Feature'] = feature_importances['Feature'].replace(feature_mapping)

non_zero_features = feature_importances[feature_importances['Importance'] > 0]
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=non_zero_features, palette="viridis")
plt.title("Najważniejsze cechy w drzewie decyzyjnym")
plt.xlabel("Waga cechy")
plt.ylabel("Cecha")
plt.show()


data = pd.DataFrame({
    "Stan cywilny: małżeństwo": X_encoded['marital-status_Married-civ-spouse']
})
counts = data["Stan cywilny: małżeństwo"].value_counts()
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis")
plt.title("Rozkład cechy: Stan cywilny - małżeństwo")
plt.xlabel("Podział na stan cywilny")
plt.ylabel("Liczba osób")
plt.xticks([0, 1], ['Nie w małżeństwie', 'W małżeństwie'])
for i, value in enumerate(counts.values):
    ax.text(i, value + 50, str(int(value)), ha='center', va='bottom', fontsize=12)
plt.show()


plt.figure(figsize=(10, 8))
sns.boxplot(x=y_encoded, y=X_encoded['education-num'])
plt.title("Interakcja: liczba lat edukacji vs dochód")
plt.xlabel("Podział na dochód")
plt.ylabel("Liczba lat edukacji")
plt.xticks([0, 1], ['Mniejszy lub równy 50K', 'Większy niż 50K'])
plt.yticks(ticks=np.arange(X_encoded['education-num'].min(), 
                           X_encoded['education-num'].max() + 0.5, 0.5))
plt.show()


clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=[feature_mapping.get(f, f) for f in X_encoded.columns],
    class_names=['<=50K', '>50K'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Drzewo decyzyjne")
plt.show()


