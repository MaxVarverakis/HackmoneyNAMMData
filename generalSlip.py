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

    def quantity(self, delX, X, Z):
            '''
            Args : 
                delX (float): amount of X to be added
                X (str): the asset (a, b, or y)
                Z (str): the asset to be swapped (a, b, or y)
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

    outer = np.linspace(1.001, 11, 100)
    inner = np.linspace(1.001, 11, 100)

    discrete_out = [i for i in range(1, 11)]
    discrete_in = [i for i in range(1, 11)]

    # ay_slippage = [d.slippage(i, 'a', 'y') for i in discrete_out]
    # by_slippage = [d.slippage(i, 'b', 'y') for i in discrete_out]
    # ab_slippage = [d.slippage(i, 'a', 'b') for i in discrete_in]
    ay_slippage = [d.slippage(i, 'a', 'y') for i in outer]
    by_slippage = [d.slippage(i, 'b', 'y') for i in outer]
    ab_slippage = [d.slippage(i, 'a', 'b') for i in inner]
    plt.plot(outer, np.log10(ay_slippage), 'r', label = 'A to Y')
    plt.plot(outer, np.log10(by_slippage), 'k:', label = 'B to Y')
    plt.plot(inner, np.log10(ab_slippage), label = 'A to B')
    plt.xlim(0, 10)
    plt.ylim(0, 2.2)
    plt.xticks([0] + discrete_out)
    plt.legend()
    plt.title('Slippage rates between assets inside and outside the nest')
    plt.xlabel('Amount of {A, B, A} being exchanged for {Y, Y, B}, respectively')
    plt.ylabel('Price Slippage (%)')
    plt.show()