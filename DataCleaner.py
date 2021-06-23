import numpy as np


class DataCleaner:
    def __init__(self):
        pass

    def _get_extreme_value(self, array_3d, cols=[1, 2], q=.95):
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
            if np.isnan(array_2d[:, i]).sum() < threshold:
                nans, f = self._nan_helper(array_2d[:, i])
                array_2d[nans, i] = np.interp(
                    f(nans), f(~nans), array_2d[~nans, i])
            else:
                inds = np.where(np.isnan(array_2d[:, i]))
                array_2d[inds, i] = np.nanmean(array_2d[:, i])

        return array_2d

    def _clean_na(self, array_3d):

        for array_2d in array_3d:
            array_2d = self._nan_interp(array_2d)

        return array_3d

    def clean_data(self, array_3d):
        X_va_cleaned = self._clean_extreme_values(array_3d)
        X_cleaned = self._clean_na(X_va_cleaned)
        return X_cleaned
