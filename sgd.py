from sklearn.linear_model import SGDClassifier
import numpy as np

"""
Diese Datei beschreibt einen Stochastic Gradient Descent (SGD) Classifier, der
Aufgrund einer (viel zu kleinen) Datenlage darauf trainiert wird zu entscheiden,
ob es sich bei einem gegebenen Objekt um einen ISP (Internet Service Provider)
handelt oder nicht.

Die verwendeten Schlüsselwörter werden in ASdb per TF-IDF Verfahren (Term
Frequency – Inverse Document Frequency) ermittelt.
"""

# Trainingsdaten
# ----------------------------------------------------------------------

# (Natürlich sind das hier viel zu wenige Daten, um einen echten, sinnvoll
# einsetzbaren Classifier zu trainieren.)
training_data = [
    {
        'keywords_count': {
            'cloud'  : 5,  'performance': 4,  'scalable': 5,
            'connect': 0,  'coverage'   : 0,  'service' : 1,
        },
        'is_isp': True
    },
    {
        'keywords_count': {
            'cloud'  : 0,  'performance': 1,  'scalable': 0,
            'connect': 5,  'coverage'   : 5,  'service' : 4,
        },
        'is_isp': False
    }
]

# Die Daten müssen zur Verwendung mit sklearn noch etwas transformiert werden.
#
# X ist hierbei eine Liste von Listen mit der Anzahl der Vorkommen der einzelnen
# Schlüsselwörter:
# [
#     [5, 4, 5, 0, 0, 1],
#     [0, 1, 0, 5, 5, 4],
#     ...
# ]
#
# y ist eine Liste der is_isp Werte: [True, False, ...]

X = list(map(lambda el: list(el['keywords_count'].values()), training_data))
y = list(map(lambda el: el['is_isp'], training_data))


# Stochastic Gradient Descent (SGD) Classifier
# ----------------------------------------------------------------------

# Da die Theorie hinter SGD Classifiern hier in der Implementierung von sklearn
# versteckt ist, empfehle ich an dieser Stelle ein Video:
# https://youtu.be/vMh0zPT0tLI

sgd_classifier = SGDClassifier() # Classifier Objekt initialisieren
sgd_classifier.fit(X, y)         # --- Classifier trainieren ---

to_predict_1 = {
    'cloud'  : 5,  'performance': 3,  'scalable': 4,
    'connect': 1,  'coverage'   : 1,  'service' : 0,
}
to_predict_2 = {
    'cloud'  : 5,  'performance': 3,  'scalable': 4,
    'connect': 20,  'coverage'   : 20,  'service' : 20,
}

to_predict_1 = np.array(list(to_predict_1.values())).reshape(1, -1)
to_predict_2 = np.array(list(to_predict_2.values())).reshape(1, -1)

print(sgd_classifier.predict(to_predict_1)) # -> True
print(sgd_classifier.predict(to_predict_2)) # -> False