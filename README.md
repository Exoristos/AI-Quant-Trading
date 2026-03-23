# AI-Quant-Trading

Günlük **BIST** ve **ABD** hisse verisinde teknik göstergeler + isteğe bağlı makro (EVDS / FRED), **PyTorch LSTM** ile yön/güven sinyali, **VectorBT** ile komisyon ve slipajlı backtest, **Streamlit** paneli.

> **Uyarı:** Araştırma ve eğitim içindir; **yatırım tavsiyesi değildir.** Backtest sonuçları veri, evren ve parametrelere bağlıdır.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Exoristos/AI-Quant-Trading/actions/workflows/ci.yml/badge.svg)](https://github.com/Exoristos/AI-Quant-Trading/actions/workflows/ci.yml)

## Özellikler

- Çok değişkenli günlük diziler üzerinde **LSTM** sınıflandırma; güven eşiği ve pozisyon yüzdesi
- **EODHD** (BIST) + **yfinance** (US) veri sağlayıcıları; makro için **TCMB EVDS** / **FRED**
- Sinyallerde **bir bar gecikme** varsayımı; maliyetlerde komisyon ve bps slipaj
- Tek sembol, BIST üyelik CSV taraması, US ticker taraması; isteğe bağlı walk-forward
- **`PUBLIC_UI`**: dağıtımda sadeleştirilmiş kenar çubuğu (ayrıntılar `.env.example`)

## Yığın

Python 3.10+, PyTorch, pandas, VectorBT, Streamlit, Plotly, pydantic-settings.

## Kurulum

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate  |  Unix: source .venv/bin/activate
pip install -e ".[dev]"
# .env.example -> .env kopyala, anahtarları doldur
streamlit run src/trading_platform/ui/app.py
```

Sadece çalıştırmak için: `pip install -r requirements.txt` (Streamlit Community Cloud bu dosyayı kullanır; paket `-e .` ile kurulur).

## Ortam değişkenleri

Şablon: [`.env.example`](.env.example). Özet:

| Değişken | Rol |
|----------|-----|
| `EODHD_API_KEY` | BIST EOD (yoksa yfinance yedeği) |
| `EVDS_API_KEY` / `FRED_API_KEY` | Makro serileri (UI’da açılır) |
| `PUBLIC_UI` | `true` ise gelişmiş kontroller gizlenir |
| `ARTIFACTS_DIR`, `BIST_MEMBERSHIP_CSV`, `MACRO_CSV_PATH` | Artifact ve dosya yolları (`BIST_MEMBERSHIP_CSV`: üyelik taraması CSV’si; tam UI’da da yol girilebilir) |

Streamlit Cloud’da aynı isimleri **Secrets** (TOML) ile verin. Ana uygulama dosyası: `src/trading_platform/ui/app.py`.

## Proje yapısı

```
src/trading_platform/
  data/      # veri, göstergeler, makro, pipeline
  models/    # LSTM, eğitim, çıkarım, walk-forward
  backtest/  # VectorBT motoru
  metrics/   # Sharpe, Sortino, drawdown (252 gün)
  ui/        # Streamlit
  config/    # AppSettings
```

## Test

```bash
pytest
```
