from UniswapSlip import uni
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

'''
Some useful formulas found at:
https://atulagarwal.dev/posts/curveamm/stableswap/
'''

class stable():
    def __init__(self, A, a, b):
        self.a = a
        self.b = b
        # self.c = c
        self.n = 2 # number of assets
        self.A = A # amplification factor
        self.D = self.d() # total number of coins when the price is the same (at equilibrium)
        self.prod = (self.D / self.n) ** self.n # product of the balance of all the coins (equivalent to x1 * x2 * ... * xn)
        self.nn = self.n ** self.n
        self.Ann = self.A * self.nn

        self.iD = self.iTot(self.a, self.b) # total number of coins when the price is the same (at equilibrium)
        self.iProd = (self.iD / self.n) ** self.n # product of the balance of all the coins (equivalent to x1 * x2 * ... * xn)
    
    def iTot(self, da, db):
        return da + db

    def iProduct(self, da, db):
        return da * db

    def iInvariant(self, da, db):
        nn = self.n ** self.n
        Ann = self.A * nn
        return Ann * self.iTot(da, db) + self.iD - Ann * self.iD - (self.iD ** (self.n + 1)) / (nn * self.iProduct(da, db))

    def tot(self, da, db):
        return (self.a + da) + (self.b - db)

    def product(self, da, db):
        return (self.a + da) * (self.b - db)


    def d(self, da = 0, db = 0):
        nn = self.n ** self.n
        Ann = self.A * nn
        d = opt.root(lambda D: Ann * self.tot(da, db) + D - Ann * D - (D ** (self.n + 1)) / (nn * self.product(da, db)), 1000)
        return d.x[0]

    # def quantity(self, db):
    #     nn = self.n ** self.n
    #     Ann = self.A * nn

    #     b = self.b + db + self.D / Ann
    #     c = self.D ** (self.n + 1) / (nn * Ann * (self.b + db))

    #     return self.a - opt.root(lambda da: da ** 2 + (b - self.D) * da - c, db).x[0]
    
    # def price(self, db):
    #     return db / self.quantity(db)

    # def slippage(self, db):
    #     return 1 - self.price(db) / self.price(1)

    def invariant(self, da, db):
        return self.Ann * self.tot(da, db) + self.D - self.Ann * self.D - (self.D ** (self.n + 1)) / (self.nn * self.product(da, db))
    
    def root(self, da):
        return opt.root(lambda db: self.Ann * self.tot(da, db) + self.D - self.Ann * self.D - (self.D ** (self.n + 1)) / (self.nn * self.product(da, db)), da).x[0]
    
    def price(self, da):
        db = self.root(da)
        return da / db

    def slippage(self, da):
        current = self.price(da)
        ref = self.price(1)
        return ref / current


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
    s = stable(100, 5, 5)
    # u = uni(100, 5, 5)

    # print(s.generalS(da = 40, db = None))
    # print(s.quantity(1))

    # print(s.generalQ(da = 1, db = None))
    x = np.linspace(1, 10, 100)
    plt.plot(x, [s.slippage(i) for i in x])
    # plt.plot(x, [s.generalS(da = i, db = None) for i in x], label = 'StableSwap Invariant')
    # plt.plot(x, [s.root(i) for i in x])
    # plt.plot(x, [s.generalQ(da = i, db = None) for i in x], label = 'StableSwap Invariant')
    plt.show()

    # delta = 2
    # xrange = np.arange(0.1, 2000, delta)
    # yrange = np.arange(0.1, 2000, delta)
    # # xrange = np.arange(0.1, 2000.0, delta)
    # # yrange = np.arange(0.1, 2000.0, delta)
    # x, y = np.meshgrid(xrange, yrange)

    # dS = np.vectorize(s.iInvariant)(x, y)
    # # dU = np.vectorize(u.invariant)(x, y)
    # plt.contour(x, y, dS, [0], colors = 'tab:blue', label = 'Stableswap Invariant')
    # plt.contour(x, y, x * y - u.k, [0], colors = 'tab:orange', label = 'Uniswap Invariant')
    # plt.legend()
    # plt.show()