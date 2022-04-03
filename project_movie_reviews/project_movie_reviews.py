# Projekt Textklassifikation

# Importe ausführen und den Datensatz laden
import numpy as np
import pandas as pd
df = pd.read_csv("moviereviews.tsv", sep="\t")

print(df.head())
print(len(df))
print(df["review"][0])
print(df["review"][2])
df["label"].value_counts()

# NaN-Werten erkennen und entfernen
print(df.isnull().sum())
df.dropna(inplace=True)
print(len(df))

# i, lb und rv für die Spalten index, label und review
blanks = list()
for i, lb, rv in df.itertuples():
    if type(rv)==str:
        if rv.isspace():
            blanks.append(i)
            
print(len(blanks))
df.drop(blanks, inplace=True)        
print(len(df)) 

# Noch einmal Blick in die Spalte "label" werfen
df["label"].value_counts()   

# Feature und Label initialisieren
X = df["review"]
y = df["label"]

# Daten in Trainig- und Testsets aufteilen
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Pipeline aufsetzen
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])
text_clf.fit(X_train, y_train)

# Modell testen und Kennzahlen ausgeben
predictions = text_clf.predict(X_test)

from sklearn import metrics
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test,predictions))

# Modell anwenden
text_clf.predict(["It's better than good. It's magnificent."])
text_clf.predict(["Craig is a towering charismatic presence from opening frame to closing shot, and he bows out in terrific, soulful, style."]) 
text_clf.predict(["Cary Joji Fukunaga has made a smashing piece of action cinema with No Time To Die - it's just a shame it had to be a Bond film. What's most disappointing is how strangely anti-climatic the whole thing feels."])
text_clf.predict(["Nice entry stars Daniel Craig as the tough, two-fisted James Bond who takes on nasty organization nicknamed Spectra with a octopus-like symbol. After a risked and disastrous assignment in Mexico, Bond is suspended. But he goes on his activities by tracking down a massive criminal syndicate. Then Bond receives a cryptic message from his dark past, it sends him pitted against a sinister and criminal group. It is led by a terrorist chief, Blofeld : Chistoph Waltz who has an ancient relationship with Bond's own childhood. While Q , Ben Wishaw, delivers him some rare and fantastic artifacts to carry out his dangerous missions. Shortly after, MI6 chief M : Ralph Fiennes is replaced by another boss, Andrew Scott. Later on, Bond meets the beautiful daughter, Lea Seydoux, of a long time enemy and then things go wrong.Once again Bond confronts an ominous and bloody organization with terrorist purports. This film takes parts of other 007 episodes as the violent fight between Bond and a hunk contender : Dave Bautista on a train , similar to Sean Connery versus Robert Shaw in From Russia with love. And the impresssive finale including the stronhold facility in the sunny desert and its destruction bears remarkable resemblance to Quantum of solace. Nicely played by Daniel Craig, this is his fourth entry, first was Casino Royale, following Quantum of solace and Skyfall. He is well accompanied by a young Bond girl, Lea Seydoux, and another Bond woman, the mature Monica Bellucci, the eldest Bond girl. The heinous leader of the powerful organization Spectra is magnetically performed by usual villain Christoph Waltz. Adding regulars of the old franchise as M well played by Ralph Fiennes, Q finely acted Ben Wishaw and Naomy Harris as Moneypenny. The film packs a colorful and glimmering cinematography by Hoyte Van Hoytema, shot in various locations as Mexico city, Austria and especially in London. As well as pounding and rousing musical score by Thomas Newman. The motion picture was well directed by Sam Mendes, though with no originaly. Mendes is a good director who has made some successful films played by important actors, such as : Jarhead, American Beauty, Revolutionary road, Road to perdition and another Bond movie : Skyfall. Rating: Above average. Well worth watching."])
text_clf.predict(["This is the worst Bond movie ever, filled with emotionless characters that I couldn't care less about. The pace of this film after a predictably exciting start is slow and boring. Unlike his fellow actors, Ben Whishaw as Q manages to portray the only believable human in this whole fake production. Why couldn't JB have been given a touch of Q's wit, humour or vulnerability? No wonder Daniel Craig wants out of this franchise - it's beneath his talent. Such a cacophony of totally forgettable dialogue, people and silly stunts is hard to imagine in a single movie and yet here it is. During one of the 'action' fights when James was being hammered by the evil assassin I noticed the person next to me had fallen asleep and was snoring. That person was an exceedingly eloquent critic."])

