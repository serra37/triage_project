# Triyaj Asistanı – Chatbot UI

Tıbbi triyaj sisteminin Streamlit tabanlı kullanıcı arayüzü.

## Özellikler

- **Hoş Geldiniz Ekranı** – İlk açılışta yönlendirici mesaj ve iki örnek prompt kartı
- **Modern Chat Arayüzü** – Kullanıcı mesajları sağda (mavi), asistan mesajları solda (beyaz)
- **Risk Sonuç Kartı** – Ön Risk Seviyesi, açıklama, önerilen yönlendirme ve güvenlik notunu içerir
- **4 Risk Seviyesi** – ACİL (kırmızı), YÜKSEK (turuncu), ORTA (sarı), DÜŞÜK (yeşil)
- **Sidebar** – Yeni kayıt butonu, geçmiş taramalar listesi, risk kılavuzu, kullanıcı kartı
- **Demo Modu** – Backend olmadan kural tabanlı yanıt üretir
- **Güvenlik Uyarısı** – Sayfanın altında sabit bar

## Çalıştırma

Önce bağımlılıkları yükleyin:

```bash
pip install streamlit
```

Uygulamayı başlatın (proje kök dizininden):

```bash
streamlit run chatbot_ui/app.py
```

## Dosya Yapısı

```
chatbot_ui/
├── app.py       # Tüm UI kodu
└── README.md    # Bu dosya
```

## Notlar

- `app.py` yalnızca `streamlit`, `uuid`, `re`, `time`, `datetime` kullanır (hepsi standart veya Streamlit bağımlılığı).
- Backend hazır olduğunda `demo_risk_analizi()` fonksiyonu yerine gerçek API çağrısı eklenebilir.
- Diğer proje dosyaları (`main.py`, `graph.py`, `agents/` vb.) **değiştirilmemiştir**.
