"""class Config:
    _instance = None
    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.settings = {}
        cls._instance.settings.update(kwargs)
        return cls._instance
c1 = Config(debug=True)
c2 = Config(url="sasdfs.com")
print(c1 is c2)
print(c1.settings, c2.settings)"""
"""
class Ayarlar:
    def __init__(self):
        print("yeni ayar objesi oluşturuldu")

a1 = Ayarlar()
a2 = Ayarlar()


class Ayarlar:
    _instance = None #sınıfın kendi içinde saklayacağı tek nesne

    def __new__(cls):
        if cls._instance is None:
            print("yeni ayar objesi oluşturuluyor...")
            cls._instance = super().__new__(cls) #1.kez oluşturuluyor...
            cls._instance.dil = "tr"
            cls._instance.header = True
        return cls._instance #her zaman aynı obje döner

    def __init__(self):
        self.tema = "white"

a1 = Ayarlar()
a2 = Ayarlar()
a1.dil = "en"
print(a1 is a2)
print(a1.dil)

class SMSBildirim:
    def gonder(self,mesaj):
        print(f"[SMS] {mesaj}")
class EmailBildirim:
    def gonder(self,mesaj):
        print(f"[Email] {mesaj}")
class WPBildirim:
    def gonder(self,mesaj):
        print(f"[WPBildirim] {mesaj}")
class BildirimFactory:
    @staticmethod
    def olustur(tip):
        if tip == "sms":
            return SMSBildirim()
        elif tip == "email":
            return EmailBildirim()
        elif tip == "whatsapp":
            return WPBildirim()
        else:
            raise ValueError("Bilinmeyen bir hata...")

b1= BildirimFactory.olustur("sms")
b1.gonder("kodunuz asdasfd")

b2 = BildirimFactory.olustur("email")
b2.gonder("asfdgdhfgdhg")

class Rapor:
    def olustur(self): pass

class PdfRapor(Rapor):
    def olustur(self):
        print("PDF oluşturuluyor...")

class ExcelRapor(Rapor):
    def olustur(self):
        print("Excel oluşturuluyor...")

class RaporFactory:
    def rapor_olustur(self):
        raise NotImplemented
class PdfRaporFactory(RaporFactory):
    def rapor_olustur(self):
        return PdfRapor()

class ExcelRaporFactory(RaporFactory):
    def rapor_olustur(self):
        return ExcelRapor()

f1 = PdfRaporFactory()
rapor1 = f1.rapor_olustur()
rapor1.olustur()

f2 = ExcelRaporFactory()
rapor2 = f2.rapor_olustur()
rapor2.olustur()



class Hamburger:
    def __init__(self, ekmek="susamlı", et=False, peynir=False, soslar=None, sebzeler=None):
        self.ekmek = ekmek
        self.et = et
        self.peynir = peynir
        self.soslar = soslar or []
        self.sebzeler = sebzeler or []

    def __repr__(self):
        return (f"hamburger(ekmek={self.ekmek}, et={self.et}, soslar={self.soslar}, sebzeler={self.sebzeler})")

class HamburgerBuilder:
    def __init__(self):
        self._ekmek = "susamlı"
        self._et = False
        self._peynir = False
        self._soslar = []
        self._sebzeler = []

    def ekmek(self,tur):
        self._ekmek = tur
        return self
    def et(self,var_mi=True):
        self._et = var_mi
        return self
    def peynir(self,var_mi=True):
        self._peynir = var_mi
        return self

    def sos(self,*isimler):
        self._soslar.extend(isimler)
        return self
    def sebze(self,*isimler):
        self._sebzeler.extend(isimler)
        return self

    def build(self):
        return Hamburger(
            ekmek=self._ekmek,
            et=self._et,
            peynir=self._peynir,
            soslar= self._soslar,
            sebzeler=self._sebzeler
        )

burger = (
    HamburgerBuilder()
    .ekmek("brioche")
    .et(True)
    .peynir(True)
    .sos("ketçap","mayonez")
    .sebze("marul","domates")
    .build()
)
print(burger)

send_email(to,subject,body) 
post_message(payload)


#target
class EmailClient:
    def send_email(self, to:str, subject:str,body:str):
        raise NotImplementedError
#adaptee ( yeni servis )
class NewMailApi:
    def post_message(self,payload:dict):
        print(f"new mail api payload= {payload}")

#adapter
class NewMailAdapter(EmailClient):
    def __init__(self,api: NewMailApi):
        self.api = api

    def send_email(self, to:str, subject:str,body:str):
        payload = {"to": to, "title": subject, "content": body}
        self.api.post_message(payload)

def sifre_sifirlama(client: EmailClient, kullaniic_maili:str):
    client.send_email(kullaniic_maili,"Şifre sıfırlama","şu linke tıkla....")

sifre_sifirlama(NewMailAdapter(NewMailApi()),"asdas@asdasd.com")

class Stok:
    def musait_mi(self,urun_id,adet): return True

class Odeme:
    def tahsil_et(self,kullanici_id, tutar): print("ödeme alındı")

class Fatura:
    def olustur(self,kullanici_id,urun_id,adet): print("fatura oluşturuldu")

class Kargo:
    def gonder(self,kullanici_id,urun_id,adet,tutar): print("kargoya verildi")

class SiparisServisi:
    def __init__(self):
        self.stok = Stok()
        self.odeme = Odeme()
        self.fatura = Fatura()
        self.kargo = Kargo()

    def siparis_ver_facede(self,kullanici_id, urun_id,adet,tutar):
        if not self.stok.musait_mi(urun_id,adet):
            print("stok yok")
            return False
        try:
            self.odeme.tahsil_et(kullanici_id,tutar)
            self.fatura.olustur(kullanici_id,urun_id,adet)
            self.kargo.gonder(kullanici_id,urun_id,adet,tutar)
            print("sipariş ok")
            return True
        except Exception as e:
            print("hata",e)
            return False

servis = SiparisServisi()
servis.siparis_ver_facede(kullanici_id=48,urun_id=2,adet=5,tutar=12)
def siparis_ver(kullanici_id,urun_id,adet,tutar):
    stok = Stok()
    odeme = Odeme()
    fatura = Fatura()
    kargo = Kargo()

    if not stok.musait_mi(urun_id,adet):
        print("stok yok")
        return
    odeme.tahsil_et(kullanici_id,tutar)
    fatura.olustur(kullanici_id,urun_id,adet)
    kargo.gonder(kullanici_id,urun_id,adet)
    print("sipariş tamamlandı...")

class FileLoader: def load(self,p): print("yükle",p)
class Parser: def parse(self,d): print("parse et", d)
class Save: def save(self,x): print("kaydet",x)

class ImportFace():
    def __init__(self):
        self.loader,self.parser,self.save = FileLoader(), Parser(), Save()
    def import_et(self,path):
        data = self.loader.load(path)
        recs = self.parser.parse(data)
        self.save.
        print(asdfds)

ImportFace().import_et("data.csv")

#Decorator
class Notifier:
    def send(self,to,msg): print("asdfsgdadfsdadfsd",msg)

class NotifWithE:
    def __init__(self, base:Notifier):
        self.base = base
    def send(self,to,msg):
        self.base.send(to,f"görsel var vs. {msg}")
n = Notifier()
a = NotifWithE(NotifWithE(Notifier())) #sarmaladık
a.send("ramazan","selam naber")

class FiyatHesaplayici:
    def __init__(self, indirim_turu, oran=0, tutar=0):
        self.indirim_turu = indirim_turu
        self.oran = oran
        self.tutar = tutar

    def hesapla(self,fiyat):
        if self.indirim_turu == "yok":
            return fiyat
        elif self.indirim_turu == "yuzde":
            return max(0, fiyat *(1- self.oran/100))
        elif self.indirim_turu == "sabit":
            return max(0, fiyat - self.tutar)
        else:
            raise ValueError("Geçersiz indirim türü")

f1 = FiyatHesaplayici("yok")
f2 = FiyatHesaplayici("yuzde",20)
f3 = FiyatHesaplayici("sabit",tutar=30)
f4 = FiyatHesaplayici("asd")

print(f1.hesapla(100))
print(f2.hesapla(100))
print(f3.hesapla(100))
print(f4.hesapla(100))

class IndirimStratejisi:
    def hesapla(self,fiyat: float) -> float:
        raise NotImplementedError

class IndirimYok(IndirimStratejisi):
    def hesapla(self,fiyat):
        return fiyat
class YuzdeIndirim(IndirimStratejisi):
    def __init__(self,yuzde):
        self.yuzde = yuzde
    def hesapla(self,fiyat):
        return max(0, fiyat *(1- self.yuzde/100))

class SabitIndirim(IndirimStratejisi):
    def __init__(self,tutar):
        self.tutar = tutar
    def hesapla(self,fiyat):
        return max(0, fiyat - self.tutar)

class BirlesikIndirim(IndirimStratejisi):
    def __init__(self,stratejiler: list[IndirimStratejisi]):
        self.stratejiler = stratejiler

    def hesapla(self,fiyat):
        sonuc = fiyat
        for s in self.stratejiler:
            sonuc = s.hesapla(sonuc)
        return sonuc
class FiyatHesaplayici:
    def __init__(self,strateji: IndirimStratejisi):
        self.strateji = strateji

    def set_strateji(self,s):
        self.stratejiler.append(s)
    def hesapla(self,fiyat):
        for s in self.stratejiler
            fiyat = s.hesapla(fiyat)
        return self.strateji.hesapla(fiyat)

def yuzde(yuzde): return lambda f: f*(1-yuzde/100)
def sabit(tutar): return lambda f: max(0,f-tutar)

def zincir(fiyat,*stratejiler):
    for s in stratejiler:
        fiyat = s(fiyat)
    return fiyat

print(zincir(200, yuzde(20), sabit(50)))
print(zincir(200, yuzde(20), sabit(50)))

fiyat = 200
indir1 = YuzdeIndirim(20)
indir2 = SabitIndirim(30)
birlesik = BirlesikIndirim([indir1,indir2])
print(birlesik.hesapla(fiyat))

h = FiyatHesaplayici(IndirimYok())
print(h.hesapla(100))

h.set_strateji(YuzdeIndirim(20))
print(h.hesapla(100))

h.set_strateji(SabitIndirim(30))
print(h.hesapla(100))

class EventBus:
    def __init__(self): self._subs = {}
    def subscribe(self,topic,fn): self._subs.setdefault(topic,[]).append(fn)
    def publish(self,topic,data):
        for fn in self._subs.get(topic,[]): fn(data)

bus = EventBus()
bus.subscribe("siparis: oluşturuldu", lambda d: print("email gitti",d))
bus.subscribe("siparis: oluşturuldu", lambda d: print("sms gitti",d))
bus.publish("siparis: oluşturuldu", {"id": 42})

ekrana, dosyaya, mobil(app)

class HavaIstasyon:
    def __init__(self):
        self.sicaklik = 25
    def guncelle(self,yeni):
        self.sicaklik = yeni

        print(f"[EKRAN] yeni sicaklik {yeni}")
        with open("log.txt","a", encoding="utf-8") as f:
            f.write(f"{yeni} C\n")
        print(f"bildirim gönderildi {yeni}")
h = HavaIstasyon()
h.guncelle(27)
h.guncelle(30)

class HavaIstasyonu:
    def __init__(self):
        self._sicaklik = 25
        self._gozlemciler = [] #observer listem

    def abone_ekle(self,obs):
        self._gozlemciler.append(obs)

    def abone_sil(self,obs):
        self._gozlemciler.remove(obs)

    def _bildir(self):
        for obs in list(self._gozlemciler):
            obs.update(self._sicaklik)
    def guncelle(self,yeni):
        self._sicaklik = yeni
        self._bildir()

class Ekran:
    def update(self,sicaklik):
        print(f"[EKRAN] yeni sicaklik {sicaklik}")
class Logger:
    def update(self,sicaklik):
        with open("log.txt","a", encoding="utf-8") as f:
            f.write(f"{sicaklik} C\n")
class MobilApp:
    def update(self,sicaklik):
        print(f"[APP] yeni sicaklik {sicaklik}")

i1 = HavaIstasyonu()
i1.abone_ekle(Ekran())
i1.abone_ekle(Logger())
i1.abone_ekle(MobilApp())

i1.guncelle(27)
i1.guncelle(29)

class Isik:
    def ac(self): print("ışık açık")
    def kapat(self): print("ışık kapalı")
class Ses:
    def __init__(self): self.seviye = 5
    def arttir(self): self.seviye += 1; print(f"ses seviyesi arttı {self.seviye}")
    def azalt(self): self.seviye -= 1; print(f"ses seviyesi azaldı {self.seviye}")

class Kumanda:
    def __init__(self):
        self.isik = Isik()
        self.ses = Ses()

    def bas(self, komut_adi):
        if komut_adi == "isik_ac":
            self.isik.ac()
        elif komut_adi == "isik_kapat":
            self.isik.kapat()
        elif komut_adi == "ses_arttir":
            self.ses.arttir()
        elif komut_adi == "ses_azalt":
            self.ses.azalt()
        else:
            print("böyle bir komut yok")
k = Kumanda()
k.bas("isik_ac")
k.bas("ses_arttir")
k.bas("ses_arttir")
k.bas("ses_azalt")
k.bas("isik_kapat")

#receiverlar :
class Isik:
    def ac(self): print("ışık açık")
    def kapat(self): print("ışık kapalı")

class Ses:
    def __init__(self): self.seviye = 5
    def arttir(self): self.seviye += 1; print(f"ses seviyesi arttı {self.seviye}")
    def azalt(self): self.seviye -= 1; print(f"ses seviyesi azaldı {self.seviye}")

#command interface
class Komut:
    def execute(self): raise NotImplementedError
    def undo(self): raise NotImplementedError

#concreatecommand
class IsikAcKomut(Komut):
    def __init__(self, isik:Isik): self.isik = isik
    def execute(self):self.isik.ac()
    def undo(self):self.isik.kapat()

class IsikKapatKomut(Komut):
    def __init__(self, isik:Isik): self.isik = isik
    def execute(self):self.isik.kapat()
    def undo(self):self.isik.ac()

class SesArttirKomutu(Komut):
    def __init__(self, ses:Ses): self.ses = ses
    def execute(self):self.ses.arttir()
    def undo(self):self.ses.azalt()

class SesAzaltKomutu(Komut):
    def __init__(self, ses:Ses): self.ses = ses
    def execute(self):self.ses.azalt()
    def undo(self):self.ses.arttir()
# Invoker
class Kumanda:
    def __init__(self):
        self._gecmis = [] #undo için stack

    def calistir(self,komut: Komut):
        komut.execute()
        self._gecmis.append(komut)

    def geri_al(self):
        if self._gecmis:
            son = self._gecmis.pop()
            son.undo()
        else:
            print("geri alınacak bir komut yok")

class MakroKomut(Komut):
    def __init__(self, komutlar: list[Komut]):
        self.komutlar = komutlar
    def execute(self):
        for c in self.komutlar:
            c.execute()
    def undo(self):
        for c in reversed(self.komutlar):
            c.undo()


isik = Isik()
ses = Ses()
ac = IsikAcKomut(isik)
kumanda = Kumanda()
kumanda.calistir(ac)

film_modu = MakroKomut([IsikAcKomut(isik),SesAzaltKomutu(ses)])
kumanda.calistir(film_modu)
kumanda.geri_al()

#kurulum kullanım (client)


kapat = IsikKapatKomut(isik)
arttir = SesArttirKomutu(ses)
azalt = SesAzaltKomutu(ses)

kumanda.calistir(arttir)
kumanda.calistir(arttir)
kumanda.calistir(arttir)
kumanda.calistir(arttir)
kumanda.geri_al() #son arttırma işlemini geri aldın.
kumanda.calistir(kapat)

class Mutfak:
    def corba_yap(self): print("mutfak çorba hazır")
    def makarna_yap(self): print("mutfak makarna hazır")
    def hesap_getir(self): print("kasa hesap hazır")
"""

def cay_hazirla():
    print("su ısıtılıyor...")
    print("çay poşetiyle demleniyor.....") #fark
    print("bardağa dökülüyor...")
    print("isteğe bağlı şeker ") #fark

def kahve_hazirla():
    print("su ısıtılıyor...")
    print("french press ile demleniyor.....") #fark
    print("bardağa dökülüyor...")
    print("isteğe bağlı süt ") #fark


from abc import ABC, abstractmethod
class IcecekHazirlama(ABC):
    def hazirla(self):
        self.su_isit()
        self.demle() #değişir
        self.bardaga_dok()
        self.ekstra() #isteğe bağlı (hook)

    #ortak adımlar alanı
    def su_isit(self): print("su ısıtılıyor...")
    def bardaga_dok(self): print("bardağa dökülüyor...")

    @abstractmethod
    def demle(self): pass

    #opsiyonel hook
    def ekstra(self): pass

class Cay(IcecekHazirlama):
    def demle(self):print("çay poşetiyle demleniyor...")
    def ekstra(self): print("şeker isteğe göre ekleniyor...")

class Kahve(IcecekHazirlama):
    def demle(self):print("french press ile demleniyor...")
    def ekstra(self): print("süt isteğe göre ekleniyor...")

class BitkiCay(IcecekHazirlama):
    def demle(self): print("kekik ile demleniyor..")
    def ekstra(self): print("karabiber ekleniyor...")
cay = Cay()
kahve = Kahve()
bitki = BitkiCay()

cay.hazirla()
kahve.hazirla()
bitki.hazirla()










