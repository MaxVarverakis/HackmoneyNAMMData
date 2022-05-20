import pandas as pd
import csv
from StableSwapSlip import stable
from UniswapSlip import uni
import random
import functools
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np

class generalSlippage():
    def __init__(self, sigma, eta, a0, b0, y0):
        self.sigma = sigma
        self.eta = eta
        self.a = a0
        self.b = b0
        self.y = y0

    def Ufun(self, a, b, y):
        x = (a ** (1 - self.sigma) + b ** (1 - self.sigma)) ** (1 / (1 - self.sigma))
        U = x ** (1 - self.eta) + y ** (1 - self.eta)
        return(U)
        
    def diff(self, da, db, dy):
        UStart = self.Ufun(self.a, self.b, self.y)
        return self.Ufun(self.a + da, self.b + db, self.y + dy) - UStart

    def manualRoots(self, **kwargs):
        '''
        Newton's method for finding root of diff function
        Must set the following parameters:
            dx = x
                the amount of x (a, b, or y) to be added
            dz = None
                z is the asset to be swapped (a, b, or y) for x

            For example, if you want to add 1 `a` to a and swap it for `y`:
                generalQ(da = 1, dy = None)
        '''
        assets = {'da': 0, 'db': 0, 'dy': 0}
        
        for key in kwargs.keys():
            if key in assets.keys() and kwargs[key] is not None:
                assets[key] = kwargs[key]
                x0 = assets[key]
            elif kwargs[key] is None:
                k = key
        # i = 0
        while abs(self.diff(**assets)) > 1e-2:
            y0 = self.diff(**assets)
            assets[k] += 1e-1
            x1 = assets[k]
            y1 = self.diff(**assets)
            deriv = 100 * (y1 - y0) / (x1 - x0)
            x0 -= y0 / deriv
            assets[k] = x0
            # print(i)
            # i += 1

        return x0.real
    
    def sameNestQuantities(self, da):
        db = (self.a ** (1 - self.sigma) + self.b ** (1 - self.sigma) - (self.a + da) ** (1 - self.sigma)) ** (1 / (1 - self.sigma)) - self.b
        return db

    def generalQ(self, **kwargs):
        '''
        Must set the following parameters:
            dx = x
                the amount of x (a, b, or y) to be added
            dz = None
                z is the asset to be swapped (a, b, or y) for x

            For example, if you want to add 1 `a` to a and swap it for `y`:
                generalQ(da = 1, dy = None)
        '''
        assets = {'da': 0, 'db': 0, 'dy': 0}

        for key in kwargs.keys():
            if key in assets.keys() and kwargs[key] is not None:
                assets[key] = kwargs[key]
        if 'da' in kwargs.keys() and kwargs.pop('da') == None:
            root = opt.root(lambda x: self.diff(x, assets['db'], assets['dy']), 3)
        elif 'db' in kwargs.keys() and kwargs.pop('db') == None:
            root = opt.root(lambda x: self.diff(assets['da'], x, assets['dy']), 3)
        elif 'dy' in kwargs.keys() and kwargs.pop('dy') == None:
            root = opt.root(lambda x: self.diff(assets['da'], assets['db'], x), 3)

        return -root.x[0]

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

    def quantity(self, delX, X, Z):
        '''
        Args
        ----
        delX : float
            amount of X to be added
        X : str
            the asset (a, b, or y)
        Z : str
            the asset to be swapped (a, b, or y)
        
        Returns how much of Z (delZ) is given when delX amount of X is put in
        '''

        UStart = self.Ufun(self.a, self.b, self.y)
        
        if X == 'a':
            delA = delX
            if Z == 'b':
                delY = 0
                delB = opt.root(lambda delB: self.Ufun(self.a + delA, self.b - delB, self.y - delY) - UStart, 3)
                return delB.x[0]
            else:
                delB = 0
                delY = opt.root(lambda delY: self.Ufun(self.a + delA, self.b - delB, self.y - delY) - UStart, 3)
                return delY.x[0]
        elif X == 'b':
            delB = delX
            if Z == 'a':
                delY = 0
                delA = opt.root(lambda delA: self.Ufun(self.a - delA, self.b + delB, self.y - delY) - UStart, 3)
                return delA.x[0]
            else:
                delA = 0
                delY = opt.root(lambda delY: self.Ufun(self.a - delA, self.b + delB, self.y - delY) - UStart, 3)
                return delY.x[0]
        elif X == 'y':
            delY = delX
            if Z == 'a':
                delB = 0
                delA = opt.root(lambda delA: self.Ufun(self.a - delA, self.b - delB, self.y + delY) - UStart, 3)
                return delA.x[0]
            else:
                delA = 0
                delY = opt.root(lambda delB: self.Ufun(self.a - delA, self.b - delB, self.y + delY) - UStart, 3)
                return delB.x[0]

    def price(self, delX, X, Z):
        delZ = self.quantity(delX, X, Z)
        price = delX / delZ

        return price
    
    def slippage(self, delX, X, Z):
        '''
        Returns the slippage in percent between delX and 1
        '''
        return (self.price(delX, X, Z) / self.price(1, X, Z) - 1) * 100

if __name__ == '__main__':
    
    d = generalSlippage(0.2, 0.8, 10, 10, 10)
    s = stable(1, 5, 5)
    u = uni(100, 5, 5)
    x = np.linspace(1, 8.25, 100)
    ax = plt.gca()
    plt.plot(x, [u.generalS(da = i, db = None) for i in x], label = 'Uniswap 100x Leverage')
    u = uni(1, 5, 5)
    plt.plot(x, [u.generalS(da = i, db = None) for i in x], label = 'Uniswap 1x Leverage')
    plt.plot(x, [s.generalS(da = i, db = None) for i in x], label = 'Stableswap Invariant: 1x amplification')
    plt.plot(1, 1, 'b', alpha = .6, label = 'Stableswap Invariant: 100x amplification')

    plt.plot(x, [d.generalS(da = i, dy = None) for i in x], label = 'A <--> Y')
    plt.plot(x, [d.generalS(da = i, db = None) for i in x], 'k', label = 'A <--> B')
    # s = stable(100, 5, 5)
    # plt.plot(x, [s.generalS(da = i, db = None) for i in x], label = 'Stableswap Invariant: 100x amplification')
    plt.tight_layout()
    plt.legend(loc = 3, prop = {'size': 20})
    plt.title('Comparing slippage rates of different invariants')
    plt.xlabel('Amount of assets being swapped')
    plt.ylabel('Price Slippage')
    plt.show()

    # d = generalSlippage(0.2, 0.8, 10, 10, 10)
    # s = stable(1, 5, 5)
    # u = uni(100, 5, 5)
    # x = np.linspace(1, 8.25, 100)
    
    # u1 = [u.generalS(da = i, db = None) for i in x]
    # u = uni(1, 5, 5)
    # u2 = [u.generalS(da = i, db = None) for i in x]
    # s = [s.generalS(da = i, db = None) for i in x]
    # d1 = [d.generalS(da = i, dy = None) for i in x]
    # d2 = [d.generalS(da = i, db = None) for i in x]

    # data = {'Uniswap 100x Leverage': u1, 'Uniswap 1x Leverage': u2, 'Stableswap Invariant: 1x amplification': s, 'A <--> Y': d1, 'A <--> B': d2}
    # df = pd.DataFrame(data)
    # df.to_csv('SlippageData.csv')

    # bools = []
    # for i in range(2):
    #     bools.append(d.manualRoots(da = i, dy = None)- d.quantity(i, 'a', 'y') < 1e-10)
    # print(np.all(bools))

    # plt.plot(x, [u.generalQ(da = i, db = None) for i in x], label = 'Uniswap : X * Y = constant')
    # plt.plot(x, [s.generalQ(da = i, db = None) for i in x], label = 'StableSwap Invariant')
    # plt.plot(x, [d.generalQ(da = i, dy = None) for i in x], label = 'A <--> Y')
    # plt.plot(x, [d.generalQ(da = i, db = None) for i in x], label = 'A <--> B')
    # plt.title('Comparing exchange rates for different market makers')
    # plt.xlabel('Amount of assets being swapped')
    # plt.ylabel('Exchange rate')
    # plt.legend()
    # plt.show()


    # ay_diffs = [d.diff(1, 0, x) for x in np.linspace(-10, 10, 100)]
    # ab_diffs = [d.diff(1, x, 0) for x in np.linspace(-10, 10, 100)]
    # plt.plot(np.linspace(-10, 10, 100), np.multiply(ay_diffs, 10))
    # plt.plot(np.linspace(-10, 10, 100), np.multiply(ab_diffs, 10))
    # plt.plot(np.linspace(-10, 10, 100), np.zeros(100), 'k:')
    # plt.show()

    # for i in range(10):
    #     print(d.sameNestQuantities(i), d.quantity(i, 'a', 'b'))

    # bools = []
    # for i in range(10):
    #     bools.append(d.manualRoots(da = i, dy = None) - d.quantity(i, 'a', 'y') < 1e-10)
    #     bools.append(d.generalP(da = i, dy = None) - d.price(i, 'a', 'y') < 1e-10)
    #     bools.append(d.generalS(da = i, dy = None) - d.slippage(i, 'a', 'y') < 1e-10)
    # print(np.all(bools))

    # outer = np.linspace(1.001, 11, 100)
    # inner = np.linspace(1.001, 11, 100)

    # discrete_out = [i for i in range(1, 11)]
    # discrete_in = [i for i in range(1, 11)]

    # # ay_slippage = [d.slippage(i, 'a', 'y') for i in discrete_out]
    # # by_slippage = [d.slippage(i, 'b', 'y') for i in discrete_out]
    # # ab_slippage = [d.slippage(i, 'a', 'b') for i in discrete_in]
    # ay_slippage = [d.slippage(i, 'a', 'y') for i in outer]
    # by_slippage = [d.slippage(i, 'b', 'y') for i in outer]
    # ab_slippage = [d.slippage(i, 'a', 'b') for i in inner]
    # plt.plot(outer, np.log10(ay_slippage), 'r', label = 'A to Y')
    # plt.plot(outer, np.log10(by_slippage), 'k:', label = 'B to Y')
    # plt.plot(inner, np.log10(ab_slippage), label = 'A to B')
    # plt.xlim(0, 10)
    # plt.ylim(0, 2.2)
    # plt.xticks([0] + discrete_out)
    # plt.legend()
    # plt.title('Slippage rates between assets inside and outside the nest')
    # plt.xlabel('Amount of {A, B, A} being exchanged for {Y, Y, B}, respectively')
    # plt.ylabel('Price Slippage (%)')
    # plt.show()