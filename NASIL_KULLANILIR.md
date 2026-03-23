# Nasıl Kullanılır? (Türkçe Kılavuz)

Bu proje, borsa verisini çekip **teknik göstergeler + LSTM** ile al/sat/bekle sinyali üretir, geçmişte bu sinyallerle **sanal alım satım (backtest)** yapar ve sonuçları **grafik + tablo** olarak gösterir. Yatırım tavsiyesi değildir; araştırma ve öğrenme içindir.

---

## Büyük resim: Veri nereden akıyor?

```text
[Veri kaynakları] → [Özellik tablosu + etiket] → [LSTM eğitimi] → [Sinyal + güven]
       → [Backtest: komisyon, slipaj, pozisyon büyüklüğü] → [Sharpe, drawdown, işlem listesi]
```

| Parça | Ne işe yarar? |
|--------|----------------|
| **Veri** | Günlük fiyat (OHLCV): ABD için çoğunlukla **yfinance**, BIST için önce **EODHD**, olmazsa **yfinance** yedeği. |
| **Makro (isteğe bağlı)** | Modele ek sütunlar: **yerel CSV**, **FRED** (ABD), **EVDS 3** (TCMB, örn. kur). Hepsi hisse takvimine **geçmişe dönük birleştirilir** (gelecek veri sızdırmamak için). |
| **Göstergeler** | RSI, MACD, Bollinger, EMA/SMA vb. — hepsi **sadece o güne kadar olan fiyatla** hesaplanır. |
| **Etiket (`y_class`)** | “Yarın (veya N gün) sonra fiyat şu kadar yukarı mı, aşağı mı, çok az mı oynadı?” → model **3 sınıf** öğrenir (al / bekle / sat). |
| **LSTM** | Geçmiş bir pencere (`seq_len` gün) özelliklerine bakıp sınıf tahmini + **güven skoru** (softmax’taki en yüksek olasılık). |
| **Güven eşiği** | Düşük güvenli tahminler **bekle** sayılır; sinyal filtrelenir. |
| **Backtest** | **VectorBT**: al/sat anları, **komisyon**, **slipaj (bps)**, pozisyonu portföyün **yüzdesi** kadar açma; sinyal bir gün **geciktirilir** (aynı kapanışta işlem yapmış gibi görünmeyelim diye). |
| **Metrikler** | Kar/zarar, Sharpe/Sortino (yıllıklandırma **252 iş günü**), max drawdown, kazanma oranı, işlem sayısı. |

---

## Kurulum (ilk sefer)

1. **Python 3.10+** kurulu olsun.
2. Proje klasöründe terminal açın:

```powershell
cd c:\Users\enise\Documents\Projelerim\lstm_entegreli
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

3. **`.env` dosyası**  
   - Kökte `.env` yoksa `.env.example` içeriğini kopyalayıp `.env` adıyla kaydedin.  
   - Anahtarları kendi hesaplarınızdan alıp doldurun (aşağıda ne oldukları yazıyor).  
   - `.env` **Git’e eklenmez** (`.gitignore`); anahtar paylaşmayın.

### `.env` içindeki anahtarlar

| Değişken | Zorunlu mu? | Ne işe yarar? |
|-----------|-------------|----------------|
| `EODHD_API_KEY` | BIST için **önerilir** | EOD Historical Data: BIST günlük barları daha düzenli çekmek için. Boşsa BIST için yfinance denenir. |
| `FRED_API_KEY` | Hayır | ABD enflasyon vb. makro (Streamlit’te “Use FRED macro” açıksa). |
| `EVDS_API_KEY` | Hayır | TCMB EVDS 3: kur / TCMB serileri (Streamlit’te “Use EVDS macro” açıksa). |
| `EVDS_BASE_URL` | Hayır | TCMB API taban adresini değiştirmeniz gerekirse. Boş bırakınca varsayılan EVDS 3 yolu kullanılır. |

---

## Arayüzü çalıştırma

```powershell
.\.venv\Scripts\activate
streamlit run src/trading_platform/ui/app.py
```

Tarayıcı açılır. Sol taraftaki **Parameters** menüsünden her şeyi seçersiniz, **Run pipeline** ile çalıştırırsınız.

---

## Çalışma modları (Run mode)

### 1) Single symbol (tek hisse)

- En basit kullanım: **bir sembol**, tarih aralığı, eğitim + backtest.
- **Market `us`:** Örn. `AAPL` — yfinance.
- **Market `bist`:**  
  - **Quick pick:** Listeden seçim (örnek likit semboller).  
  - **Custom:** Kendi yazdığınız kod (ör. `THYAO` veya `THYAO.IS` — sistem `.IS` ekler).

**Train new model:** İşaretliyse her koşuda o veri için model yeniden eğitilir (daha uzun sürer). Kapatırsanız, **Artifacts directory** altında daha önce kaydedilmiş `meta.json` + `lstm_classifier.pt` + `scaler.npz` olmalı.

### 2) Scan: membership CSV (BIST)

- **Birden fazla BIST sembolü** için aynı pipeline’ı tekrarlar (her birine ayrı model, ayrı klasör).
- **BIST membership CSV:** `rebalance_date,ticker` sütunları. Aynı tarihte birden fazla satır = o tarihten itibaren geçerli endeks/sepet.  
  Şablon: `examples/bist30_membership_template.csv`  
- **End date:** Bu tarihe kadar geçerli olan **son rebalance** listesi alınır; tarama o listeden yapılır.
- **Max symbols per scan:** Aynı anda en fazla kaç sembol (zaman ve donanım için sınır).

Sonuç: Tablo — sembol başına getiri, Sharpe, drawdown, işlem sayısı, hata durumu.

### 3) Scan: US tickers (comma list)

- Virgülle ayrılmış ABD sembolleri (örn. `AAPL,MSFT,GOOG`).  
- Yine **max symbols** ile sınırlanır.

---

## Önemli parametreler (kısa sözlük)

| Ayar | Anlamı |
|------|--------|
| **Start / End date** | Geçmiş veri penceresi. |
| **Initial capital** | Sanal başlangıç parası. |
| **Commission (fraction)** | İşlem ücreti oranı (0.001 = %0,1). |
| **Slippage (bps)** | Kayma; 5 bps ≈ fiyatın %0,05’i kadar maliyet varsayımı. |
| **LSTM confidence threshold** | Model “emin değilim” diyorsa sinyal **bekle** olur. |
| **Position size (% of portfolio)** | Her girişte portföy değerinin yaklaşık yüzde kaçı ile işe girilsin (VectorBT **target percent** mantığı). |
| **Sequence length** | Modele kaç günlük geçmiş pencere verileceği. |
| **Training epochs** | Tek sembol modunda eğitim tur sayısı (fazla = daha uzun, aşırı = ezber riski). |
| **Scan: epochs per symbol** | Tarama modunda sembol başına daha kısa eğitim için. |
| **Use FRED macro** | `FRED_API_KEY` ile ABD makro sütunları eklenir. |
| **Use EVDS macro** | `EVDS_API_KEY` ile TCMB serileri eklenir. **EVDS series codes:** virgülle birden fazla kod (örn. `TP.DK.USD.A.YTL`). |
| **Macro CSV path** | Kendi CSV’niz: mutlaka **`date`** adında bir tarih sütunu olsun; diğer sütunlar sayısal makro olabilir. Örnek: `examples/macro_sample.csv`. |
| **Label horizon (bars)** | Kaç gün ilerideki getiriye göre “al/sat/bekle” etiketi üretileceği. |
| **HOLD band ε** | İlerideki getiri mutlak değeri bu kadar küçükse etiket **bekle** (gürültüyü azaltır). |
| **Artifacts directory** | Tek sembol modelinin kaydedileceği klasör (örn. `artifacts`). |
| **Scan artifacts root** | Tarama modunda her sembol alt klasöre yazılır (`artifacts/scan/THYAO_IS` gibi). |
| **Walk-forward analysis** | Sadece **Single symbol** sonunda: zamanı dilimlere bölüp, her dilimde train/val metrikleri tablosu (LSTM’in zamanda ne kadar tutarlı göründüğüne dair fikir; yine de tek doğrulama yöntemi değil). |

---

## Proje klasörü (kafayı karıştırmaması için)

```text
src/trading_platform/
  data/          → veri çekme, makro, göstergeler, tarama
  models/        → LSTM, eğitim, tahmin, walk-forward
  strategies/    → güven eşiği, sinyal → al/sat giriş çıkış
  backtest/      → VectorBT motoru, komisyon/slipaj
  metrics/       → Sharpe, Sortino, drawdown vb.
  ui/            → Streamlit arayüzü
  config/        → .env ayarları (pydantic-settings)
examples/        → örnek CSV’ler, BIST üyelik şablonu
tests/           → otomatik testler
```

---

## Komut satırı: test

```powershell
pytest
```

---

## Sık karşılaşılan sorunlar

| Sorun | Ne yapmalı? |
|--------|-------------|
| BIST veri gelmiyor | `.env` içinde `EODHD_API_KEY` dolu mu? Tarih aralığı ve sembol (`.IS`) doğru mu? |
| EVDS boş / hata | Seri kodu EVDS 3 arayüzündekiyle aynı mı? Gerekirse `EVDS_BASE_URL` ile TCMB’nin güncel API yolunu yazın. |
| Makro CSV yok sayılıyor | Yol **tam dosya yolu** veya proje kökünden göreli doğru mu? `date` sütunu var mı? |
| Eğitim çok yavaş | Epoch düşürün, tarih aralığını kısaltın, `seq_len` küçültün; taramada **max symbols** düşük tutun. |
| Walk-forward hata veriyor | Veri çok kısa olabilir; tarih aralığını uzatın veya split sayısını azaltın. |

---

## Dürüst uyarılar

- **Geçmiş performans** geleceği göstermez.  
- **BIST30 şablonu** örnek listedir; gerçek tarihsel endeks üyeliği için kendi `membership` CSV’nizi oluşturmalısınız; aksi halde **survivorship bias** olur.  
- Günlük bar backtest, gün içi fiyat yolunu bilmez; slipaj ve komisyon **basit modeldir**.  
- Makro serilerinde **yayın tarihi** ile **gözlem dönemi** karışmamalı; ciddi çalışmada veri sözlüğünü kontrol edin.

---

## İngilizce kısa README

Kurulum özeti ve İngilizce notlar için kökteki [`README.md`](README.md) dosyasına bakabilirsiniz.
