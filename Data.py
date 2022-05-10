'''
1. Create Math equations that gives us price of y in terms of a
2. Create a loop to iterate through and simulate different slippage paths
3. Graph
'''

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

'''
sigma = 0.2
eta = 0.8

Ufun = function(a,b,y) {
  
  x = (a^(1-sigma) + b^(1-sigma))^(1/(1-sigma))
  U = x^(1-eta) + y^(1-eta)
  
  return(U)
  
}

a0 = 10
b0 = 10
y0 = 10

Ustart = Ufun(a0, b0, y0)
Ufun(a0, b0, y0)

ayprice = function(delA) {
  
  rootfun = function(delY) {
    return(Ufun(a0 + delA, b0, y0 - delY) - Ustart)
  }
  
  
  out = uniroot(rootfun, c(0, 10))$root
  return(out)
  
}
'''

class data():
    def __init__(self, sigma, eta, a0, b0, y0):
        self.sigma = sigma
        self.eta = eta
        self.a = a0
        self.b = b0
        self.y = y0

    def Ufun(self, a, b, y):
        x = (a**(1-self.sigma) + b**(1-self.sigma))**(1/(1-self.sigma))
        U = x**(1-self.eta) + y**(1-self.eta)
        return(U)

    def quantity(self, delA):
        UStart = self.Ufun(self.a, self.b, self.y)
        delY = opt.root(lambda delY: self.Ufun(self.a + delA, self.b, self.y - delY) - UStart, 3.11)

        return delY
    
    def price(self, delA):
        delY = self.quantity(delA).x[0]
        ratio = delA / delY


        return ratio

    def slippage(self, delA1, delA2):
        return (self.price(delA1) / self.price(delA2) - 1) * 100

d = data(.2, .8, 100, 100, 100)
UStart = d.Ufun(d.a, d.b, d.y)
# delY = d.quantity(3)
# print(delY.x[0])
# print(d.quantity(3).x[0])
# print(d.price(260))
# print(d.slippage(260, 259))
delA = [i for i in range(1000)]
slippage = [d.slippage(i, i - 1) for i in delA]
plt.plot(delA, slippage)
plt.ylim([0, 1])
plt.xlim([0, 1000])
plt.xlabel('Amount of A')
plt.ylabel('Slippage (%)')
plt.show()

# price = [d.price(i) for i in delA]
# plt.plot(delA, price)
# plt.show()