# AI-Quant: Derin Öğrenmeli Algoritmik İşlem ve Backtest Motoru

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Specify-lightgrey) <!-- SPDX rozeti ve LICENSE dosyası hazır olunca güncelleyin -->

---

## Özet

**AI-Quant**, **BIST** ve **ABD** günlük bar verileri üzerinde sistematik hisse işlemi için uçtan uca bir **araştırma ve simülasyon** yığınıdır. Platform, çok değişkenli ve **nedensel** sinyaller (yön ve güven) üreten **PyTorch LSTM** katmanını, **vektörel VectorBT** simülasyonu ile birleştirir. Tasarımda **zaman sırası disiplini** (aynı barda işlem yok; sinyaller bir bar gecikmeli), **maliyet farkındalıklı PnL** (komisyon ve slipaj) ve yalnızca fiyata dayanmayan **makro zengin özellik setleri** ön plandadır. **Streamlit** arayüzü eğitim, backtest ve tanılamayı tek akışta toplar; hem teknik iterasyon hem paydaş incelemesi için uygundur.

> **Uyarı:** Bu yazılım araştırma ve eğitim içindir. **Yatırım tavsiyesi değildir.** Geçmiş backtest performansı—aşağıdaki örnek metrikler dahil—gelecek getirinin göstergesi değildir; veri kalitesi, evren tanımı ve hiperparametrelere bağlıdır.

---

## Temel özellikler — farkı ne?

- **Derin öğrenme çekirdeği**  
  Çok değişkenli günlük diziler **PyTorch LSTM**’e girer; tahmin tarzı sinyal üretimi `src/trading_platform/models/` altında yapılandırılmış eğitim ve çıkarım yoluyla entegredir.

- **Makroekonomik bağlam**  
  OHLCV’nin ötesinde boru hattı **TCMB EVDS** serilerini (ör. **USD/TRY** ve ilgili Türkiye makro kodları) ve isteğe bağlı **FRED** verisini alabilir; takvim hizasında yanlışlıkla **ileriye bakmayı** azaltmak için katı **geriye dönük** `merge_asof` hizalaması kullanılır.

- **Gerçekçi backtest**  
  **VectorBT** motoru **komisyon**, **slipaj** ve **dinamik hedef maruziyeti** (`targetpercent` / yüzde tabanlı boyutlandırma) modeller. Sinyaller, ürettikleri bara göre **bir bar yürütme gecikmesi** ile uygulanır; günlük veride kapanış–ertesi iş günü açılışı tarzı realismaya yaklaşır.

- **Etkileşimli panel**  
  **Streamlit** ve **Plotly** ile **özsermaye eğrileri**, **underwater (drawdown)** grafikleri ve **işlem günlükleri** incelenir; tek sembol, **BIST üyelik CSV taraması** ve **ABD ticker listesi** modları vardır. Risk metrikleri (ör. **Sharpe**, **Sortino**, **maksimum drawdown**) uygun yerlerde **252** iş günü ile annualize edilir.

---

## Örnek stres testi (BIST oynaklığı)

Son dönemde, yüksek oynaklıklı bir **BIST** hissesi (**THYAO**) üzerinde **stres tarzı** bir konfigürasyonda yığın, **yüksek güven, düşük frekans** politikasını andıran **disiplinli risk–getiri** davranışı göstermiştir:

| Boyut | Gösterge sonuç |
|--------|----------------|
| **Kazanma oranı** | ~**%91** (işlem bazında, konfigürasyona bağlı) |
| **Sharpe oranı** | **> 1,5** (günlük getiriler üzerinden annualize, 252 gün) |
| **Maksimum drawdown** | Test edilen maliyet ve boyut varsayımları altında **<%5** |

Bu sonuçlar, **güven eşikli sinyal politikası** ve **maliyet modelinin** gürültülü rejimlerde **katılımı baskılayıp** yine de uygun hareketleri yakalayabildiğine dair kanıt olarak okunmalıdır; canlı performans vaadi değildir. Doğrulama için arayüzde desteklenen **walk-forward** analizi ve **örnek dışı (out-of-sample)** bölünmeleri kullanın.

---

## Teknoloji yığını

| Katman | Kütüphaneler ve notlar |
|--------|-------------------------|
| Dil | **Python** ≥ 3.10 |
| Derin öğrenme | **PyTorch**, **scikit-learn** |
| Veri ve sayısal işlem | **pandas**, **numpy**, **requests** |
| Piyasa verisi | **yfinance**; BIST EOD kalitesi için isteğe bağlı **EODHD** |
| Backtest | **VectorBT** |
| Arayüz ve grafik | **Streamlit**, **Plotly** |
| Yapılandırma | **pydantic**, **pydantic-settings**, **python-dotenv** |

Bağımlılık sürümleri ve ek paketler **`pyproject.toml`** içinde tanımlıdır.

---

## Kurulum ve kullanım

### 1. Klon ve sanal ortam

```bash
git clone <YOUR_REPO_URL> lstm_entegreli
cd lstm_entegreli
python -m venv .venv
```

Sanal ortamı etkinleştirin:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

### 2. Bağımlılıkları kurma

Proje **PEP 621** (`pyproject.toml`) kaynağıdır; repoda sabit bir `requirements.txt` yoktur. Paketi **editable** modda, geliştirici araçlarıyla (**pytest**) kurun:

```bash
pip install -e ".[dev]"
```

*Kurumunuz düz bir kilit dosyası istiyorsa, temiz bir ortamdan dışa aktarın (örn. `pip freeze > requirements.txt`) ve `pip install -r requirements.txt` kullanın; sürümler `pyproject.toml` ile uyumlu kalmalıdır.*

### 3. API anahtarları (`.env`)

Şablonu kopyalayıp anahtarları doldurun:

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Unix benzeri
```

| Değişken | Amaç |
|----------|------|
| `EODHD_API_KEY` | **BIST** EOD için önerilir; yoksa **yfinance** yedeği |
| `EVDS_API_KEY` | **TCMB EVDS 3** makro (örn. USD/TRY); arayüzden açın |
| `EVDS_BASE_URL` | İsteğe bağlı (varsayılan: `https://evds3.tcmb.gov.tr/service/evds/`) |
| `FRED_API_KEY` | İsteğe bağlı **FRED** makro serileri; arayüzden açın |

`.env` **gitignore**’dadır; gizli anahtarları repoya göndermeyin.

### 4. Uygulamayı çalıştırma

```bash
streamlit run src/trading_platform/ui/app.py
```

### 5. Testler

```bash
pytest
```

---

## Proje yapısı (özet)

Kaynak kod **`src/trading_platform/`** altında: `data/` (sağlayıcılar, EVDS, makro, göstergeler, pipeline), `models/` (LSTM, train/inference, walk-forward), `strategies/`, `backtest/`, `metrics/`, `ui/`, `config/`. Çıktılar varsayılan olarak **`artifacts/`** altına gider. Operasyonel notlar (Streamlit modları, EVDS/FRED dayanıklılığı, backtest semantiği) için **[HANDOFF.md](HANDOFF.md)** dosyasına bakın.

---

## Ek okuma

- **Türkçe ayrıntılı kullanım:** [NASIL_KULLANILIR.md](NASIL_KULLANILIR.md)  
- **Katkı / oturum devri özeti:** [HANDOFF.md](HANDOFF.md)

---

## Araştırma uyarıları (bir kez okuyun)

- Tarihsel testlerde tarihli evren dosyası olmadan bugünkü endeks üyeliği kullanılırsa **survivorship bias** riski.  
- **Makro revizyonları** ve **yayın gecikmeleri** tam modellenmemiştir; makro birleştirmeleri kehanet özelliği değil hizalama yardımı olarak görün.  
- **Günlük barlar** gün içi yolu yansıtmaz; slipaj ve ücretler stilize yaklaşımlardır.  
- **Kazanma oranı** tek başına yeterli istatistik değildir; maliyetlerle birlikte **beklenti**, **drawdown** ve **ciro**yu birlikte raporlayın.
