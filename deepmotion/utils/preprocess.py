__author__ = 'Christian Dansereau'

import numpy as np
from sklearn import linear_model

class ConfoundsRm:
    def __init__(self, confounds, data, intercept=True):
        self.fit(confounds, data, intercept)

    def fit(self, confounds, data, intercept=True):
        self.data_dim = data.shape
        if confounds == []:
            print('No confounds')
            self.nconfounds = 0
        else:
            if len(self.data_dim) == 3:
                self.a1, self.a2, self.a3 = data.shape
                data_ = data.reshape((self.a1, self.a2 * self.a3))
            elif len(self.data_dim) == 4:
                self.a1, self.a2, self.a3, self.a4 = data.shape
                data_ = data.reshape((self.a1, self.a2 * self.a3 * self.a4))
            else:
                data_ = data
            self.nconfounds = confounds.shape[1]
            self.reg = linear_model.LinearRegression(fit_intercept=intercept)
            # print data_.shape,confounds.shape
            self.reg.fit(confounds, data_)

    def transform(self, confounds, data):
        # compute the residual error
        if self.nconfounds == 0:
            return data
        else:
            if len(data.shape) == 3:
                data_ = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
                res = data_ - self.reg.predict(confounds)
                return res.reshape((data.shape[0], data.shape[1], data.shape[2]))
            elif len(data.shape) == 4:
                data_ = data.reshape((data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))
                res = data_ - self.reg.predict(confounds)
                return res.reshape((data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
            else:
                data_ = data
                return data_ - self.reg.predict(confounds)

    def transform_batch(self, confounds, data, batch_size=50):
        # compute the residual error
        if self.nconfounds == 0:
            return data
        else:
            # batch convert the data
            nbatch = data.shape[0] / (batch_size)  # number of batch
            batch_res = []
            for idx_batch in range(nbatch):
                if idx_batch == nbatch - 1:
                    batch_res.append(
                        self.transform(confounds[idx_batch * batch_size:-1, ...], data[idx_batch * batch_size:-1, ...]))
                else:
                    batch_res.append(self.transform(confounds[idx_batch * batch_size:(1 + idx_batch) * batch_size, ...],
                                                    data[idx_batch * batch_size:(1 + idx_batch) * batch_size, ...]))
            return np.vstack(batch_res)

    def nConfounds(self):
        return self.nconfounds

    def intercept(self):
        if len(self.data_dim) == 3:
            return self.reg.intercept_.reshape((1, self.data_dim[1], self.data_dim[2]))
        elif len(self.data_dim) == 4:
            return self.reg.intercept_.reshape((1, self.data_dim[1], self.data_dim[2], self.data_dim[3]))
        else:
            return self.reg.intercept_


