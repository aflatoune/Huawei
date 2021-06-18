## Défi du réseau d'accès optique

## Préparation des données

Les données brutes sont situées dans le sous-dossier `/data`. Elles consistent en un ensemble de quelques milliers de séries temporelles qui représentent environ 20 jours de temps.

```
data/
    ville_A/
        source.npy
        source_labels.npy
    city_B/
        target.npy
        target_labels.npy
        test.npy
        test_labels.npy
```

Cependant, le défi consiste à prédire l'échec en utilisant seulement une seule semaine de données.
Par conséquent, nous pré-traitons les données originales avec le code suivant

```
~/hackathon/data $ python prepare_data.py
```
Ce programme prendra en compte les données originales et produira un fichier `pickle` qui générera des sous-séries temporelles d'une durée d'une semaine.
La sortie sera située comme suit
```
data/
    ville_A/
        ramp_train.pickle
    ville_B/
        ramp_target.pickle
        ramp_test.pickle
```
Ces fichiers sont ceux qui seront lus par `problem.py`.

Attention : nous vous laissons les données originales pour que vous puissiez les explorer. Mais l'ensemble des données privées a été généré en utilisant le programme original `prepare_data.py`.

Alors faites-en ce que vous voulez, nous n'oublions pas comment les données privées sont traitées.