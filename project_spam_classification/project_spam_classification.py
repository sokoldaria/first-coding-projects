# Bibliotheken laden
import numpy as np
import pandas as pd

# Datei einlesen; tsv steht für tab separated; sep in diesem Beispiel ist der Tab
df = pd.read_csv("smsspamcollection.tsv", sep="\t")


# Den Datensatz erkunden
print(df.head())
print(df.tail())
print(len(df))
print(df.isnull().sum()) # Keine Nullwerte vorhanden; ggfl. die fehlenden Werte mit dem Mittelwert ausfüllen

# Eindeutige Werte ausgeben
df["label"].unique() 
# Anzahl der eindeutigen Werte ausgeben
df["label"].value_counts()
# Wir sehen, dass 4825 von 5572 Nachrichten, oder 86.6%, ham sind.
# Das bedeutet, dass das von uns anvisierte Modell des maschinellen Lernens über 86.6% korrekt sein muss, um besser als eine Zufallsentscheidung zu sein.

# Daten visualisieren
import matplotlib.pyplot as plt

# Info zur Spalte "length" ausgeben
df["length"].describe()

plt.xscale("log")
bins = 1.15**(np.arange(0,50))
plt.hist(df[df["label"]=="ham"]["length"], bins=bins, alpha=0.8) # alpha für Transparenz
plt.hist(df[df["label"]=="spam"]["length"], bins=bins, alpha=0.8)
plt.legend(("ham", "spam"))
plt.show()
# Es sieht aus, als gäbe es einen kleinen Bereich von Werten, bei dem eine Nachricht mit höherer Wahrscheinlichkeit spam als ham ist.
# --> Bei einer Länge von ca. 300 Zeichen ist es oft ein Spam.

# Info zur Spalte "punct" ausgeben
df["punct"].describe()

plt.xscale("log")
bins = 1.15**(np.arange(0,15))
plt.hist(df[df["label"]=="ham"]["punct"], bins=bins, alpha=0.8) # alpha für Transparenz
plt.hist(df[df["label"]=="spam"]["punct"], bins=bins, alpha=0.8)
plt.legend(("ham", "spam"))
plt.show()
# Dies sieht sogar noch schlechter aus - es gibt anscheinend keine Werte, die eine Entscheidung Richtung spam oder ham ermöglichen.
# Wir werden trotzdem versuchen, eine Klassifizierung mit maschinellem Lernen vorzunehmen,
# aber wir müssen schlechte Ergebnisse erwarten.

# Features und Label definieren
X = df[["length", "punct"]] # Sub-Dataframe
print(X.head())

y = df["label"]
print(y.head())


# Die Daten in Trainig- und Testsets aufteilen
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # random_state für die gleiche Aufteilung

print(X_train)
print(X_train.shape)

print(X_test)
print(X_test.shape)

print(y_train)
print(y_train.shape)

print(y_test)
print(y_test.shape)


# Einen Klassifizierer für logistische Regression trainieren
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(solver="lbfgs") # solver zum Mischen (shuffle) der Daten auswählen
lr_model.fit(X_train, y_train)


# Das Modell testen und Kennzahlen ausgeben
from sklearn import metrics
predictions = lr_model.predict(X_test)

print(metrics.confusion_matrix(y_test, predictions))
# --> Kein gutes Ergebnis!
print(metrics.classification_report(y_test, predictions))
# --> Ham kann man mit dem Modell relativ gut vorhersagen, jedoch Spam nicht!
# Precision-Score für alle (alle Vorhersagen aufgerechnet)
print(metrics.accuracy_score(y_test, predictions))
# --> Dieses Modell ist schlechter, als wenn einfach alle Nachrichten als "ham" deklariert werden!

# Einen Naiven Bayes trainieren
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Das Modell testen und Kennzahlen ausgeben
predictions = nb_model.predict(X_test) # predictions -> "y Dach"

print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))
# --> Besser, aber immer noch schlechter als 86.6%


# Einen Support Vector Machine (SVM) Klassifizierer trainieren
from sklearn.svm import SVC
svc_model = SVC(gamma='auto') # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
svc_model.fit(X_train,y_train)

# Das Modell testen und Kennzahlen ausgeben
predictions = svc_model.predict(X_test)

print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))
# --> Und endlich haben wir ein Modell, dass ein wenig besser funktioniert als die Zufallsauswahl.