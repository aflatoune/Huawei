import numpy as np
import datetime
import pandas as pd
import time


class FeatureExtractor:
    def __init__(self):
        self.count = 0

    def transform(self, X):
        if not (len(X) == 29592) and (not len(X) == 50862) and (not len(X) == 47275):
            if self.count >= 1:
                self.count = 1
            cleaner = DataCleaner(drop_olt_recv=True)
            X = cleaner.clean_data(X)
            prep = PrepareExtractor()
            X, _ = prep.get_data(X,
                                 col_names=cleaner.columns,
                                 size_sample=-1,
                                 resample={'func': 'mean', 'unit': 'H'},
                                 slice_=[(12, 36), (-24, None)],
                                 name=self.count)

            self.count += 1
        print(X.shape)
        return X


class DataCleaner:
    """
    Class to remove extreme values and NAs from the initial dataset.
    """

    def __init__(self, drop_olt_recv=True):
        self.drop_olt_recv = drop_olt_recv
        if self.drop_olt_recv:
            self.columns = ["current",
                            "err_down_bip",
                            "err_up_bip",
                            "rdown",
                            "recv",
                            "rup",
                            "send",
                            "temp",
                            "volt"]
        else:
            self.columns = ["current",
                            "err_down_bip",
                            "err_up_bip",
                            "olt_recv",
                            "rdown",
                            "recv",
                            "rup",
                            "send",
                            "temp",
                            "volt"]

    def _get_extreme_value(self, array_3d, cols=[1, 2], q=.9):
        sub_array_3d = array_3d[:, :, cols]
        array_2d = np.concatenate([obs for obs in sub_array_3d], axis=0)
        extreme_val = np.nanquantile(array_2d, q=q, axis=0)
        return {k: v for k, v in zip(cols, extreme_val)}

    def _clean_extreme_values(self, array_3d):
        d = self._get_extreme_value(array_3d)

        for array_2d in array_3d:
            for col in d.keys():
                inds = np.where(array_2d[:, col] > d[col])
                array_2d[inds, col] = d[col]

        return array_3d

    def _nan_helper(self, array_1d):
        """
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices=index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        """

        return np.isnan(array_1d), lambda z: z.nonzero()[0]

    def _nan_interp(self, array_2d, threshold=12):

        for i in range(array_2d.shape[1]):
            if 0 < np.isnan(array_2d[:, i]).sum() <= threshold:
                nans, f = self._nan_helper(array_2d[:, i])
                array_2d[nans, i] = np.interp(
                    f(nans), f(~nans), array_2d[~nans, i])
            elif np.isnan(array_2d[:, i]).sum() > threshold:
                inds = np.where(np.isnan(array_2d[:, i]))
                array_2d[inds, i] = np.nanmean(array_2d[:, i])
            else:
                pass

        return array_2d

    def _clean_na(self, array_3d):

        for array_2d in array_3d:
            array_2d = self._nan_interp(array_2d)

        return array_3d

    def clean_data(self, array_3d):
        start = time.time()
        if self.drop_olt_recv:
            array_3d = np.delete(array_3d, 3, axis=2)

        X_va_cleaned = self._clean_extreme_values(array_3d)
        X_cleaned = self._clean_na(X_va_cleaned)
        print("Temps clean : ", str(time.time() - start))
        return X_cleaned


class PrepareExtractor:
    def __init__(self):
        """
        - create_df_obs : permet de cr??er pour une observation donn??es
                        un dataframe de l'ensemble des donn??es temporelles.
                        Les fonctionnalit??s suivantes sont pr??sentes :
                            - `get_unit` permet de r??cup??rer les indications temporelles
                                (jour, heure, minute)
                            - une fonction de resampling de date sous la forme
                                'unit': unit?? d'aggr??gation ??? D (jour), T (minute), H (heure)
                                avec la possibiit?? de choisir une fr??quence (6H, toutes
                                                                            les 6 heures)
                                'func': m??thode d'aggr??gation : moyenne, max, minimum
        - flatten_df : permet d'applatir le dataframe cr??e par `create_df_obs`
                    afin de le transformer en une observation
        - get_data : permet de cr??er une dataframe de plusieurs observations
                    ?? partir des ??tapes `create_df_obs` et `flatten_df`
        - concat_Xy : permet de concat??ner le dataframe cr??er par `get_data`
                    avec ses labels `y`
        """
        pass

    def get_data(self,
                 X,
                 col_names=["current",
                            "err_down_bip",
                            "err_up_bip",
                            "olt_recv",
                            "rdown",
                            "recv",
                            "rup",
                            "send",
                            "temp",
                            "volt"],
                 y=None,
                 size_sample=1000,
                 resample={'unit': 'D', 'func': 'mean'},
                 source='source',
                 random_state=1,
                 name=None,
                 verbose=False,
                 first_diff=False,
                 add_unit=[],
                 slice_=0,
                 fast=True):

        np.random.seed(random_state)

        if not isinstance(X, np.ndarray):
            if size_sample == -1:
                sample = range(len(getattr(X, source)))
            else:
                sample = np.random.choice(range(len(getattr(X, source))),
                                          replace=False,
                                          size=size_sample)
        else:
            if size_sample == -1:
                sample = range(len(X))
            else:
                sample = np.random.choice(range(len(X)),
                                          replace=False,
                                          size=size_sample)

        liste_X = []
        if y is not None:
            array_y = np.zeros(len(sample))

        start = time.time()
        for p, i in enumerate(sample):
            X_, y_ = self.create_df_obs(X=X,
                                        col_names=col_names,
                                        y=y,
                                        source=source,
                                        sample=i,
                                        slice_=slice_,
                                        add_unit=add_unit,
                                        resample=resample,
                                        first_diff=first_diff,
                                        verbose=verbose)
            if fast:
                X_ = self.flatten(X_, name=name)
            else:
                X_ = self.flatten_df(X_, name=name)

            liste_X.append(X_)

            if y is not None:
                array_y[p] = y_
            else:
                array_y = None

        print("Temps prepare : ", str(time.time() - start))
        if fast:
            X = np.vstack(liste_X)
        else:
            X = pd.concat(liste_X)
        return X, array_y

    def concat_Xy(self, X, y):
        X["target"] = y
        return X

    def flatten(self, X, name=None):
        if name is None:
            return X.to_numpy().reshape(1, -1)
        X = X.to_numpy().reshape(1, -1)
        return np.append(X, name)

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

    def create_df_obs(self,
                      X,
                      col_names=["current",
                                 "err_down_bip",
                                 "err_up_bip",
                                 "olt_recv",
                                 "rdown",
                                 "recv",
                                 "rup",
                                 "send",
                                 "temp",
                                 "volt"],
                      y=None,
                      source='source',
                      sample=44,
                      first_diff=0,
                      add_unit=[],
                      slice_=0,
                      resample=None,
                      verbose=False):
        """
        Permet de r??cup??rer une observation et son label
        sur une source sp??ficique de donn??es. Possibilit??
        de :
            - choisir un resampling temporelle
            - ajouter des variables temporelles
        Param??tres
            - X : OpticalDataset
            - y : OpticalLabels
            - source : d??faut 'source', possible valeurs :
                        [source', 'source_bkg','target',
                        'target_bkg', 'target_unlabeled']
            - extra : par d??faut '', ajoute un id aux colonnes
            - add_unit : Ajoute une colonne temporelle parmi
                        ["day", "hour", "minute"], par d??faut toutes
            - sample : int, par d??faut 44, num??ro de l'observation
            - resample : d??faut None, requiert un dictionnaire :
                        {"unit": '<unit?? de temps>',
                        "func": '<fonction d'aggr??gration>'}
            - verbose : affiche les actions effectu??es
        Return
            X, y : pd.Dataframe de l'observation et son label
        """
        if y is not None:
            if isinstance(y, np.ndarray):
                y = y[sample]
                pass
            elif source in ['source', 'target']:
                y = getattr(y, source)[sample]
            else:
                y = None

        # permet de cr??er une datetime 1 janvier 2000 ?? 00:00 et
        # selectionne l'observation (feature x unit?? de temps).
        # Fixe ensuite un index de 15 min entre chaque unit??

        start = datetime.datetime(2000, 1, 1, 0)
        if not isinstance(X, np.ndarray):
            obs = getattr(X, source)[sample]
        else:
            obs = X[sample]

        index = pd.date_range(start, periods=len(obs), freq="15T")

        X = pd.DataFrame(obs, index=index,
                         columns=col_names)

        if resample:
            X = self._resampling(X=X, resample=resample,
                                 columns=col_names, verbose=verbose)
        if first_diff:
            X = self.add_first_diff(X, n_diff=first_diff)

        if slice_:
            len_index = len(X)
            slice_ = self.get_slicing(len_index=len_index, liste_borne=slice_)
            X = X.iloc[slice_, :]

        if add_unit:
            add_unit = self._get_add_unit(add_unit)
            X = X.apply(self.get_unit, **add_unit, axis=1)
        return X, y

    def get_slicing(self, len_index, liste_borne):
        slicing_ = []
        for sl in liste_borne:
            if len(sl) == 2:
                s, e = sl
                p = 1
            else:
                s, e, p = sl
            slicing_ += list(range(len_index))[s:e][::p]
        return slicing_

    def add_first_diff(self, X, n_diff):
        X = pd.concat([X[n_diff:], X.diff(n_diff)[n_diff:]], axis=1)
        return X

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
        Cr??e une nouvelle variable ?? partir de l'index de temps d'un df.
        Les valeurs sont des unit??s de temps et non des
        jours/heures/minutes r??elles.
        Param??tres
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
