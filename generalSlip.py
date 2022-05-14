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

    def generalQ(self, **kwargs):
        '''
        Must set the following variables:
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
        Returns the slippage in percent between kwarg 1 and kwarg 2
        '''
        
        current = self.generalP(**kwargs)
        kwargs[list(kwargs.keys())[0]] = 1
        ref = self.generalP(**kwargs)

        return (current / ref - 1) * 100

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

# def testU(**kwargs):
#         d = generalSlippage(0.2, 0.8, 10, 10, 10)
#         UStart = d.Ufun(d.a, d.b, d.y)

#         assets = {'a': 0, 'b': 0, 'y': 0}

#         for key in kwargs.keys():
#             if key in assets.keys():
#                 assets[key] = kwargs[key]
#             else:
#                 assets[key] = key

#         return opt.root(d.Ufun(d.a + assets['a'], d.b + assets['b'], d.y + assets['c']) - UStart)

# def partial(*kwargs):
#     f = functools.partial(testU, delta)
#     for delta in kwargs:
#         f = functools.partial(f, delta)
    # return functools.partial(testU, delta)

if __name__ == '__main__':
    d = generalSlippage(0.2, 0.8, 10, 10, 10)
    
    bools = []
    for i in range(10):
        bools.append(d.generalQ(da = i, dy = None) - d.quantity(i, 'a', 'y') < 1e-5)
        bools.append(d.generalP(da = i, dy = None) - d.price(i, 'a', 'y') < 1e-5)
        bools.append(d.generalS(da = i, dy = None) - d.slippage(i, 'a', 'y') < 1e-5)
    print(np.all(bools))

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