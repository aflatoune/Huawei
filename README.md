# Défi du réseau d'accès optique

# Introduction

Le réseau d'accès optique (OAN) est une solution courante de réseau d'accès domestique à large bande dans le monde entier. Il relie les abonnés des terminaux à leur fournisseur de services. Les défaillances du réseau affectent à la fois la qualité du service (QoS) et l'expérience de l'utilisateur (la qualité d'expérience QoE). Pour réduire les dommages, il est important de prévoir à l'avance les défaillances du réseau et de les réparer à temps. Les algorithmes d'apprentissage machine (ML) ont été largement utilisés comme solution pour construire ces modèles de prédiction des pannes. 

Cependant, la plupart des modèles d'apprentissage automatique sont spécifiques aux données et ont tendance à se dégrader lorsque la distribution des données change. Le premier défi de données de Huawei France de cette année vise à résoudre ce problème. 

Vous recevrez un ensemble de données étiquetées sur le réseau d'accès optique d'une ville que nous appelons "A" (que nous appelons le domaine source) et un ensemble de données pour la plupart non étiquetées d'une ville "B" (que nous appelons le domaine cible).

On vous demande de construire une solution d'apprentissage par transfert en utilisant les données sources étiquetées et les données cibles non étiquetées pour entraîner un modèle de prédiction de panne pour la ville B. Il s'agit d'un **problème d'adaptation de domaine non supervisée (UDA)**. Pour être précis, nous incluons un petit nombre de points cibles étiquetés dans l'ensemble d'entraînement, de sorte que nous pouvons appeler cette configuration "UDA à quelques coups" ou "adaptation de domaine semi-supervisée".


1. **valeurs manquantes** : il y a beaucoup de valeurs manquantes dans les données ;
2. **séries temporelles de données de capteurs** ;
3. **déséquilibre des classes** : les défaillances du réseau sont rares, il s'agit donc d'un problème de classification très déséquilibré. 


## Contexte

Les technologies de transmission ont évolué pour intégrer les technologies optiques jusque dans les réseaux d'accès, au plus près de l'abonné. Actuellement, la fibre optique est le support de transmission par excellence en raison de sa capacité à propager le signal sur de longues distances sans régénération, de sa faible latence et de sa très grande largeur de bande. La fibre optique, initialement déployée dans les réseaux à très longue distance et à très haut débit, tend aujourd'hui à se généraliser pour offrir des services plus grand public en termes de bande passante. Il s'agit des technologies FTTH pour "Fiber to the Home ".

Le FTTH généralement adopté par les opérateurs est une architecture PON (Passive Optical Network). Le PON est une architecture point à multipoint basée sur les éléments suivants :
- Une infrastructure de fibre optique partagée. L'utilisation de coupleurs optiques dans le réseau est la base de l'architecture et de l'ingénierie de déploiement. Les coupleurs sont utilisés pour desservir plusieurs zones ou plusieurs abonnés.


- Equipement central faisant office de terminaison de ligne optique (OLT). L'OLT gère la diffusion et la réception des flux à travers les interfaces du réseau. Il reçoit les signaux des abonnés et diffuse un contenu basé sur des services spécifiques. 


- Équipements terminaux :
    - ONT (Optical Network Terminations) dans le cas où l'équipement est dédié à un client et que la fibre atteint le client. Il s'agit alors d'une architecture de type FTTH (Fiber To The Home). Il n'y a qu'une seule fibre par client (les signaux sont bidirectionnels).
    - ONU (Optical Network Unit) dans le cas où l'équipement est dédié à un bâtiment entier. Il s'agit alors d'une architecture de type FTTB (Fiber To The Building).

<img src="https://image.makewebeasy.net/makeweb/0/p4Ky6EVg4/optical%20fiber-knowledge/Apps_FTTx_Fig3.png">

Les données pour ce défi sont collectées à partir de capteurs au niveau de l'ONT.

### Les données

Les données proviennent de deux villes différentes : la ville A (la source) et la ville B (la cible). Les données sont étiquetées pour la ville A mais (principalement) non étiquetées pour la ville B (seulement 20% des données étiquetées sont connues pour la ville B). Pour les deux villes A et B, les données sont une série temporelle collectée pendant environ 60 jours. La granularité de la série temporelle est de 15 minutes. Les échantillons représentent différents utilisateurs (donc différents ONT). A chaque pas de temps, nous disposons d'une mesure en dix dimensions des caractéristiques suivantes (entre parenthèses, les unités de chaque caractéristique).

- **current** : courant de polarisation du module optique de l'ONT GPON (mA)
- **err_down_bip** : nombre de trames descendantes ONT avec erreur BIP (entier)
- **err_up_bip** : nombre de trames ONT amont avec erreur BIP (entier)
- **olt_recv** : puissance de réception du module optique GPON ONT de l'ONU (dBm)
- **rdown** : débit descendant de l'ONT GPON (Mbs)
- **recv** : puissance de réception du module optique GPON ONT (dBm)
- **rup** : débit amont de l'ONT GPON (Mbs)
- **send** : puissance d'émission du module optique GPON ONT (dBm)
- **temp** : température du module optique GPON ONT (Celsius)
- **volt** : tension d'alimentation du module optique GPON ONT (mV)
- **étiquettes** : 0 (faible) ou 1 (échec) pour l'échantillon. 

L'objectif du défi est de séparer le faible de l'échec, les bonnes données sont juste données comme information secondaire (pouvant être utilisées pour la calibration), ainsi l'objectif est de soumettre un classificateur binaire.

Soit $x_t$ l'échantillon collecté au jour $t$, alors l'étiquette correspondante est calculée au jour $t+7$. Notre objectif est de prédire un échec à partir de données provenant de 7 jours auparavant.


Les données sont données avec la forme **[users, timestamps, features]** et les features sont données dans le même ordre que celui présenté ci-dessus. Pour chaque utilisateur et chaque horodatage, nous agrégeons sept jours de données.

Notez que l'ensemble de données publiques (qui vous est remis avec le kit de démarrage) et l'ensemble de données privées (utilisé pour évaluer vos soumissions sur le serveur) proviennent de la même distribution, donc en principe vous pourriez utiliser les données cibles publiques étiquetées pour apprendre un classificateur et soumettre la fonction réelle. Cela irait à l'encontre de l'objectif de l'apprentissage par transfert, nous avons donc décidé de transformer légèrement mais significativement l'ensemble de données privées pour rendre cette stratégie non performante.

### Métriques

- Accuracy (**acc**): Le nombre d'étiquettes correctement prédites par rapport au nombre total d'échantillons.  [sklearn function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score). 
- Area unther the ROC curve (**auc**). Ce score nous donne la probabilité qu'une instance d'échec soit mieux notée qu'une instance faible par la fonction discriminante binaire [sklearn function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
- Average precision (**ap**): il résume une courbe précision-rappel sous la forme de la moyenne pondérée des précisions obtenues à chaque seuil, l'augmentation du rappel par rapport au seuil précédent étant utilisée comme poids [sklearn function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score).
- **Precision@Recall**: est un score hybride implémenté dans `utils.scores`. Il calcule la précision lorsque le rappel est à un certain pourcentage, c'est-à-dire, recall est la précision lorsque le rappel est à k%.

**NOTE : Average precision (ap) est la métrique officiel d'évaluation**.


#### Données manquantes

Vous remarquerez que certaines données sont manquantes dans les ensembles de données. Il peut y avoir plusieurs raisons :

1. Aucune donnée n'a été collectée à une date spécifique pour un utilisateur spécifique.
2. Le processus de collecte des données ne parvient pas à récupérer une caractéristique.

Cela fait partie du défi de surmonter cette difficulté de la vie réelle.

Pour installer `ramp-workflow`:
```
pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
```

Cette commande installera la bibliothèque `rampwf` et le script `ramp-test` que vous pouvez utiliser pour vérifier votre soumission avant de la soumettre. Vous n'avez pas besoin de connaître ce paquetage pour participer au défi, mais il pourrait être utile de jeter un coup d'œil à la [documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/index.html) si vous souhaitez savoir ce qui se passe lorsque nous testons votre modèle, en particulier la page [exécution RAMP](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/scoring.html) pour comprendre `ramp-test`, et les [commandes](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/command_line.html) pour comprendre les différentes options de la ligne de commande. 


Vous pouvez regarder le code du flux de travail à `external_imports/utils/workflow.py` pour voir exactement comment vos soumissions sont chargées et utilisées. Vous pouvez exécuter l'entraînement et la prédiction de votre soumission ici dans le notebook. Lorsque vous exécutez `ramp-test`, nous faisons une validation croisée ; ici vous utilisez les données complètes de formation pour former et les données de test pour tester. [Cette page](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/scoring.html) vous donne un bref aperçu de ce qui se passe en coulisses lorsque vous exécutez le script `ramp-test`.

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