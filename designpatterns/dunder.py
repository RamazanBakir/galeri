"""class Basket:
    def __init__(self,items):
        self.items = items #[(ürünadı,fiyat),....]

    def __len__(self):
        return len(self.items)

    def total(self):
        return sum(price for _, price in self.items)
    def __eq__(self, other):
        return self.total() == other.total()

sepet1 = Basket([("elma",20),("armut",30),("ki",60),("muz",10)])
sepet2 = Basket([("elma",20),("ki",90),("muz",10)])

print(sepet1.total())
print(sepet2.total())
print(sepet1 == sepet2)
print(len(sepet1))
"""