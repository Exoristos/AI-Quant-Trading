# Proje handoff (yeni chat için)

Kısa bağlam: **Python** ile BIST/US günlük veri → teknik göstergeler → **PyTorch LSTM** (al/bekle/sat + güven) → **VectorBT** backtest (komisyon, slipaj, hedef % pozisyon) → **Streamlit + Plotly** arayüz. Yatırım tavsiyesi değildir.

---

## Çalıştırma

```powershell
cd c:\Users\enise\Documents\Projelerim\lstm_entegreli
.\.venv\Scripts\activate
pip install -e ".[dev]"
streamlit run src/trading_platform/ui/app.py
pytest
```

---

## Ortam (`.env`)

| Anahtar | Not |
|---------|-----|
| `EODHD_API_KEY` | BIST EOD için önerilir; yoksa yfinance yedeği |
| `FRED_API_KEY` | İsteğe bağlı makro (UI’da FRED checkbox) |
| `EVDS_API_KEY` | TCMB EVDS 3; UI’da EVDS checkbox |
| `EVDS_BASE_URL` | Varsayılan: `https://evds3.tcmb.gov.tr/service/evds/` — TCMB değiştirirse |

`.env` `.gitignore`’da; şablon: `.env.example`.

---

## Dizin yapısı (`src/trading_platform/`)

| Klasör | İçerik |
|--------|--------|
| `data/` | `providers/` (yfinance, EODHD), `macro.py`, `evds.py`, `indicators.py`, `pipeline.py`, `scan.py`, `bist_membership.py`, `bist_universe.py` |
| `models/` | LSTM, `train.py`, `inference.py`, `walk_forward.py`, `dataset.py` |
| `strategies/` | Güven eşiği, sinyal → giriş/çıkış |
| `backtest/` | VectorBT motoru, maliyetler |
| `metrics/` | Sharpe/Sortino/MDD (252 gün) |
| `ui/` | `app.py` — Streamlit (yalnızca **İngilizce** UI; TR i18n geri alındı) |
| `config/` | Pydantic settings (`AppSettings`) |

Örnek dosyalar: `examples/`, Türkçe kullanım: `NASIL_KULLANILIR.md`.

---

## Streamlit modları

1. **Single symbol** — tek hisse, eğitim + backtest + grafikler  
2. **Scan: membership CSV (BIST)** — `rebalance_date,ticker` CSV; `tickers_as_of(end_date)`  
3. **Scan: US tickers** — virgülle liste  

Artefaktlar: `artifacts/` (tek sembol), `artifacts/scan/...` (tarama).

---

## Önemli davranışlar / düzeltmeler

- **EVDS/FRED:** Sunucu bazen **HTTP 200 + HTML** döndürüyor; `evds.py` / `macro.py` içinde JSON değilse **boş DataFrame** + `logger.warning` (tarama artık `Expecting value` ile çökmez).  
- **Backtest:** Sinyaller **1 bar gecikmeli** (aynı kapanışta işlem yok varsayımı).  
- **Pozisyon:** VectorBT `targetpercent` (fallback: `percent`).  
- **Walk-forward:** Tek modda checkbox; `models/walk_forward.py`.

---

## Teknik yığın

`pyproject.toml`: pandas, numpy, yfinance, requests, torch, scikit-learn, vectorbt, streamlit, plotly, pydantic, pydantic-settings, python-dotenv.

---

## Yeni chat’te örnek prompt

> Repo: `lstm_entegreli`. Özet: `HANDOFF.md`. Şunu yap: [özellik/hata]. İlgili dosyalar: `src/trading_platform/ui/app.py`, `data/pipeline.py`, …

---

*Son güncelleme: handoff dosyası; UI dili İngilizce.*
