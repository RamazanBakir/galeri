"""def logla(*args):
    metin = " | ".join(str(x) for x in args)
    print(f"[LOG] {metin}")

logla("kullanıcı girişi: ","ramazan","başarılı")
logla("dosya açıldı: ","config.yaml","başarılı","afdassd","asda")

def baglani_ayarlari(**kwargs):
    sunucu = kwargs.get("host","localhost")
    port = kwargs.get("port",3306)
    ssl = kwargs.get("ssl",False)
    print(f"sunucu: {sunucu}, Port: {port}, {ssl}")

baglani_ayarlari(host="app.blabla.com", port=5153)
@property

class Rectangle:
    def __init__(self, width: float, height:float):
        self.width = width
        self.height = height

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self,value: float):
        value = float(value)
        if value <= 0 :
            raise ValueError("genişlik pozitif olması gerekiyor...")
        self._width = value

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float):
        value = float(value)
        if value <= 0:
            raise ValueError("yükseklik pozitif olması gerekiyor...")
        self._height = value
    @property
    def area(self) -> float:
        return self._width * self._height #sadece okunur olacak

    @property
    def cevre(self) -> float:
        return 2 * (self._width + self._height)

r = Rectangle(3,4)
print(r.area)
print(r.cevre)
r.width = 100
r.height = -1
"""

