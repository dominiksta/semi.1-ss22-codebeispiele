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
        'keywords': {
            'cloud'  : True,  'performance': True,  'scalable': True,
            'connect': False, 'coverage'   : False, 'service' : False,
        },
        'is_isp': True
    },
    {
        'keywords': {
            'cloud'  : False, 'performance': False, 'scalable': False,
            'connect': True,  'coverage'   : True,  'service' : True,
        },
        'is_isp': False
    }
]

# Die Daten müssen zur Verwendung mit sklearn noch etwas transformiert werden.
#
# X ist hierbei eine Liste von Listen mit True/False Werten zu den einzelnen
# Schlüsselwörtern:
# [
#     [True, False, True, ...],
#     [False, False, True, ...],
#     ...
# ]
#
# y ist eine Liste der is_isp Werte: [True, False, ...]

X = list(map(lambda el: list(el['keywords'].values()), training_data))
y = list(map(lambda el: el['is_isp'], training_data))


# Stochastic Gradient Descent (SGD) Classifier
# ----------------------------------------------------------------------

# Da die Theorie hinter SGD Classifiern hier in der Implementierung von sklearn
# versteckt ist, empfehle ich an dieser Stelle ein Video:
# https://youtu.be/vMh0zPT0tLI

sgd_classifier = SGDClassifier() # Classifier Objekt initialisieren
sgd_classifier.fit(X, y)         # --- Classifier trainieren ---

to_predict = np.array(
    list({
        'cloud'  : False, 'performance': False, 'scalable': False,
        'connect': True,  'coverage'   : True,  'service' : False,
    }.values())
).reshape(1, -1)

print(sgd_classifier.predict(to_predict))