import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

class uni():
    def __init__(self, k, a, b):
        self.a = a
        self.b = b
        self.k = k * self.a * self.b

    def invariant(self, da, db):
        '''
        x * y = k = constant
        '''
        return (self.a + da) * (self.b - db) - self.k
    
    def generalQ(self, **kwargs):
        assets = {'da': 0, 'db': 0}
        
        for key in kwargs.keys():
            if key in assets.keys() and kwargs[key] is not None:
                assets[key] = kwargs[key]
                x0 = assets[key]
            elif kwargs[key] is None:
                k = key
        while abs(self.invariant(**assets)) > 1e-2:
            y0 = self.invariant(**assets)
            assets[k] += 1e-1
            x1 = assets[k]
            y1 = self.invariant(**assets)
            deriv = 100 * (y1 - y0) / (x1 - x0)
            # print(deriv)
            x0 -= y0 / deriv
            assets[k] = x0

        return x0.real

    def generalP(self, **kwargs):
        '''
        Returns the price between kwarg 1 and kwarg 2
        '''
        price = list(kwargs.values())[0] / self.generalQ(**kwargs)
        return price

    def generalS(self, **kwargs):
        '''
        Returns the slippage in decimal between kwarg 1 and kwarg 2
        '''
        
        current = self.generalP(**kwargs)
        kwargs[list(kwargs.keys())[0]] = 1
        ref = self.generalP(**kwargs)

        # return (current / ref - 1) * 100
        return ref / current

if __name__ == '__main__':
    u = uni(1, 10, 10)
    # print(u.quantities(da = 1, db = None))
    x = np.linspace(1, 8, 100)
    # plt.plot(x, [u.generalQ(da = i, db = None) for i in x])
    plt.plot(x, [u.generalS(da = i, db = None) for i in x])
    plt.show()