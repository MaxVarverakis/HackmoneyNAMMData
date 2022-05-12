import scipy.optimize as opt
import matplotlib.pyplot as plt

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
    def testQuantity(self, delX, X, Z):
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
            
    def testPrice(self, delX, X, Z):
        delZ = self.testQuantity(delX, X, Z)
        price = delX / delZ

        return price
    
    def testSlippage(self, delX1, delX2, X, Z):
        '''
        Returns the slippage in percent between delA1 and delA2
        '''
        return (self.testPrice(delX1, X, Z) / self.testPrice(delX2, X, Z) - 1) * 100