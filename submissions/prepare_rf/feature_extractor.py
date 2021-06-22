import numpy as np
import datetime
import pandas as pd
import time


class FeatureExtractor:
    def __init__(self):
        pass

    def transform(self, X):
        np.nan_to_num(X, copy=False)
        prep = PrepareExtractor()
        X, y = prep.get_data(X, size_sample=len(X), resample={'unit': 'D', 'func': 'mean'})
        return X


class PrepareExtractor:
    def __init__(self):
        """
        - create_df_obs : permet de créer pour une observation données
                        un dataframe de l'ensemble des données temporelles.
                        Les fonctionnalités suivantes sont présentes :
                            - `get_unit` permet de récupérer les indications temporelles
                                (jour, heure, minute)
                            - une fonction de resampling de date sous la forme
                                'unit': unité d'aggrégation – D (jour), T (minute), H (heure)
                                avec la possibiité de choisir une fréquence (6H, toutes
                                                                            les 6 heures)
                                'func': méthode d'aggrégation : moyenne, max, minimum
        - flatten_df : permet d'applatir le dataframe crée par `create_df_obs`
                    afin de le transformer en une observation
        - get_data : permet de créer une dataframe de plusieurs observations
                    à partir des étapes `create_df_obs` et `flatten_df`
        - concat_Xy : permet de concaténer le dataframe créer par `get_data`
                    avec ses labels `y`
        """
        pass

    def get_data(self, X_train, y_train=None,
                size_sample=1000,
                resample={'unit': 'D', 'func': 'mean'},
                source='source',
                random_state=1,
                name=None):

        np.random.seed(random_state)

        if not isinstance(X_train, np.ndarray):
            sample = np.random.choice(range(len(getattr(X_train, source))),
                                    replace=False,
                                    size=size_sample)
        else:
            sample = np.random.choice(range(len(X_train)), replace=False, size=size_sample)

        liste_X = []
        array_y = np.zeros(len(sample))

        start = time.time()
        for p, i in enumerate(sample):
            X, y = self.create_df_obs(X=X_train,
                                y=y_train,
                                source=source,
                                sample=i,
                                add_unit=[],
                                resample=resample,
                                verbose=False)
            X = self.flatten_df(X, name=name)
            liste_X.append(X)
            array_y[p] = y

        print("Temps : ", str(time.time() - start))
        X = pd.concat(liste_X)

        return X, y

    def concat_Xy(self, X, y):
        X["target"] = y
        return X

    def flatten_df(self, X, name=None):
        add_unit = ['day', 'hour', 'minute']
        col_names = []
        for i in X.index:
            for c in X.columns:
                if c in add_unit:
                    col_names.append(c)
                else:
                    col_names.append(f'{i} {c}')
        tmp = pd.DataFrame(X.stack(dropna=False).values).T
        tmp.columns = col_names
        if name is not None:
            tmp["groupe"] = name
        return tmp

    def create_df_obs(self, X, y=None,
                      source='source',
                      sample=44,
                      extra='',
                      add_unit=["day", "hour", "minute"],
                      resample=None,
                      verbose=True):
        """
        Permet de récupérer une observation et son label
        sur une source spéficique de données. Possibilité
        de :
            - choisir un resampling temporelle
            - ajouter des variables temporelles

        Paramètres
            - X : OpticalDataset
            - y : OpticalLabels
            - source : défaut 'source', possible valeurs :
                        [source', 'source_bkg','target',
                        'target_bkg', 'target_unlabeled']
            - extra : par défaut '', ajoute un id aux colonnes
            - add_unit : Ajoute une colonne temporelle parmi
                        ["day", "hour", "minute"], par défaut toutes
            - sample : int, par défaut 44, numéro de l'observation
            - resample : défaut None, requiert un dictionnaire :
                        {"unit": '<unité de temps>',
                        "func": '<fonction d'aggrégration>'}
            - verbose : affiche les actions effectuées

        Return
            X, y : pd.Dataframe de l'observation et son label
        """
        if y is not None:
            if source in ['source', 'target']:
                y = getattr(y, source)[sample]
            else:
                y = None

        # permet de créer une datetime 1 janvier 2000 à 00:00 et
        # selectionne l'observation (feature x unité de temps).
        # Fixe ensuite un index de 15 min entre chaque unité

        start = datetime.datetime(2000, 1, 1, 0)
        if not isinstance(X, np.ndarray):
            obs = getattr(X, source)[sample]
        else:
            obs = X[sample]

        index = pd.date_range(start, periods=len(obs), freq="15T")

        columns = ["current",
                   "err_down_bip",
                   "err_up_bip",
                   "olt_recv",
                   "rdown",
                   "recv",
                   "rup",
                   "send",
                   "temp",
                   "volt"]

        columns = [c + extra for c in columns]

        X = pd.DataFrame(obs, index=index,
                         columns=columns)

        if resample:
            X = self._resampling(X=X, resample=resample, columns=columns, verbose=verbose)

        add_unit = self._get_add_unit(add_unit)
        X = X.apply(self.get_unit, **add_unit, axis=1)
        return X, y

    def _resampling(self, X, resample, columns, verbose=True):
        if verbose:
            print(f'Resampling with {resample}')

        if isinstance(resample, list):
            liste_r = []
            for r in resample:
                if 'unit' in r and 'func' in r:
                    s_X = getattr(X.resample(r["unit"]), r["func"])()
                    s_X.columns = [c + '-' + r['func'] for c in columns]
                    liste_r.append(s_X)

                else:
                    raise KeyError("Invalid key must have {'unit', 'func'}")
            X = pd.concat(liste_r, axis=1)

        else:
            if 'unit' in resample and 'func' in resample:
                X = getattr(X.resample(resample["unit"]), resample["func"])()
            else:
                raise KeyError("Invalid key must have {'unit', 'func'}")
        return X

    def _get_add_unit(self, add_unit):
        params = {"day": False,
                  'hour': False,
                  'minute': False}

        for i in add_unit:
            if i in params:
                params[i] = True
        return params

    def get_unit(self, row, day=True, hour=True, minute=True):
        """
        Crée une nouvelle variable à partir de l'index de temps d'un df.
        Les valeurs sont des unités de temps et non des
        jours/heures/minutes réelles.

        Paramètres
            - day : par defaut True, ajoute le jour
            - hour : par defaut True, ajoute l'heure
            - minute : par defaut True, ajoute la minute

        """
        if day:
            row["day"] = 'D' + str(row.name.day)
        if hour:
            row["hour"] = 'H' + str(row.name.hour)
        if minute:
            row['minute'] = 'M' + str(row.name.minute)
        return row
