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

    def slippage(self, delA):
        return (((self.price(delA) / self.price(1)) - 1) * 100)

    # def limit(self, delA):
    #   y0 = self.y
    #   y0 = y0 - self.quantity(delA).x[0]
    #   if int(y0) > 0:
    #     print(1) 
    #   else:
    #     print('More liquidity must be locked in asset Y before swap can be executed.')


d = data(.2, .8, 10, 10, 10)
UStart = d.Ufun(d.a, d.b, d.y)
# print(d.limit())
# delY = d.quantity(3)
# print(delY.x[0])
# print(d.price(3))
# d.quantity(2).x[0]
# print(d.quantity(3).x[0])
# print(d.quantity(35).x[0])
# print(d.slippage(3, 1))
# print(d.slippage(5))
# print((d.price(3)/d.price(1) - 1) * 100)

'''
Graph of incremental Y received for 1 unit increases in A.
'''
delA = [i for i in range(100)]
difference = [(d.quantity(i).x[0]-d.quantity(i-1).x[0]) for i in range(100)]
# difference = np.log10(np.array(difference))
plt.plot(delA, difference)
plt.ylim([0, 1])
plt.xlim([0, 12])
plt.xlabel('Amount of A added to LP')
plt.ylabel('Incremental Amount of Y Received')
plt.show()



'''
Graph of price slippage
'''
delA = [i for i in range(100)]
slippage = ([d.slippage(i) for i in range(100)])
slippage = np.log10(np.array(slippage))
plt.plot(delA, slippage)
plt.xlim(0,35)
plt.ylim(0, 2.2)
plt.xlabel('Amount of A being exchanged for Y')
plt.ylabel('Price Slippage of Expected Price of Y versus Actual (%)')
plt.show()

'''
slippage as a function of price

how much A you spend in order to get 1 y should be
increasing, while the quantity of A increases
and while the quantity of y decreases.
'''
slippage =np.array([d.slippage(i) for i in range(100)])
price = np.array([d.price(i) for i in range(100)])
delA = [i for i in range(100)]
# price = np.log10(price)
slippage = np.log10(slippage)
plt.plot(delA, slippage)
plt.xlim(0,10)
plt.ylim(0, 5)
plt.xlabel('Quantity of A in LP')
plt.ylabel('Price of A per Y, log10')
plt.show()

# price = [d.price(i) for i in delA]
# plt.plot(delA, price)
# plt.show() 


class data_ab():
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
        delB = opt.root(lambda delB: self.Ufun(self.a + delA, self.b - delB, self.y) - UStart, 3.44)

        return delB

    def price(self, delA):
        delB = self.quantity(delA).x[0]
        ratio = delA / delB

        return ratio

    def slippage(self, delA):
        return ((self.price(delA) / self.price(1)) - 1) * 100

b = data_ab(.2, .8, 10, 10, 10)
UStart = b.Ufun(b.a, b.b, b.y)
UStart

# delB = b.quantity(2)
# print(delB.x[0])
# print(b.quantity(3).x[0])
# print(b.quantity(35).x[0])
print(b.slippage(5))
print(b.price(3))
b.quantity(2).x[0]

'''
Graph of amount of b received per 1 unit increase in A
'''

delA = [i for i in range(100)]
difference = [(b.quantity(i).x[0]-b.quantity(i-1).x[0]) for i in range(100)]
plt.plot(delA, difference)
plt.ylim([0, 2])
plt.xlim([0, 11])
plt.xlabel('Amount of A added to LP')
plt.ylabel('Amount of B Received per 1 Unit Increase in A')
plt.show()

'''
Graph of price slippage for AB swaps

how much A you spend in order to get 1 B should be
increasing, while the quantity of A increases
and while the quantity of b decreases.
'''
price = np.array([b.price(i) for i in range(100)])
delA = [i for i in range(100)]
slippage = np.array([b.slippage(i) for i in range(100)])
slippage = np.log10(slippage)
plt.plot(delA, slippage)
plt.xlim(0,10)
plt.ylim(0, 5)
plt.xlabel('Amount of A being exchanged for B')
plt.ylabel('Price Slippage of Expected Price of B versus Actual (%)')
plt.show()


#calculating percentage change between price slippages
num = (np.array([d.slippage(i) for i in range(10)]) - np.array([b.slippage(i) for i in range(10)]))
print(num)
perc = num/np.array([b.slippage(i) for i in range(10)])
print(perc[2:])
print(np.average(perc))
