"""
Paper Trading Bot — Çoklu Coin Tarama + Riske Göre Bütçe Dağıtımı
====================================================================
Her çalıştığında (GitHub Actions ile örn. 15 dakikada bir):
  1. Binance'te en yüksek 24 saatlik hacme sahip ilk N coini bulur
  2. Her biri için RSI + MA crossover sinyali VE volatilite (risk) hesaplar
  3. AL sinyali veren coinler arasında, sermayeyi DÜŞÜK volatiliteye daha
     çok, YÜKSEK volatiliteye daha az pay verecek şekilde dağıtır
     (risk-paritesi mantığı: risk ne kadar yüksekse, o kadar az bütçe)
  4. Halihazırda pozisyonda olan coinler için SAT sinyali / stop-loss /
     take-profit kontrolü yapar
  5. Durumu (cash + açık pozisyonlar) state.json'a kaydeder
"""

import ccxt
import pandas as pd
import numpy as np
import json
import csv
import os
import urllib.request
import urllib.parse
from datetime import datetime, timezone

# =====================================================================
# AYARLAR
# =====================================================================
# Binance ve KuCoin, GitHub Actions'ın çalıştığı bulut sunucu IP'lerini
# (genelde ABD/Azure) kısıtlı bölge sayıp erişimi engelliyor (451 hatası).
# Kraken, ABD'de zaten lisanslı/uyumlu çalıştığı için bu tür IP'leri
# engellemiyor — bu yüzden varsayılan olarak Kraken kullanılıyor.
# Kraken'de çoğu parite USDT yerine USD ile işlem görüyor.
EXCHANGE_ID = "kraken"
QUOTE_CURRENCY = "USD"
TOP_N_COINS = 100            # Hacme göre taranacak coin sayısı
TIMEFRAME = "15m"
CANDLE_LIMIT = 50            # Sinyal/volatilite hesabı için çekilecek mum sayısı

STARTING_BALANCE_USD = 2000.0

# --- Risk / bütçe dağıtım kuralları ---
MAX_CONCURRENT_POSITIONS = 8      # Aynı anda en fazla kaç coinde pozisyon olsun
MAX_ALLOC_PER_COIN_PCT = 0.25     # Tek coine, toplam bütçenin en fazla %'si kadar yatırılsın
MIN_TRADE_USD = 15                # Bu tutarın altındaki işlemler açılmaz (anlamsız ufalanmayı önler)
CASH_RESERVE_PCT = 0.10           # Bakiyenin bu kadarı hiç yatırılmadan nakit tutulsun

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MA_FAST = 9
MA_SLOW = 21

STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.05

# Stablecoin'ler (dolara sabitli) volatilitesi neredeyse sıfır olduğu için
# risk-ağırlıklı dağıtım yanlışlıkla en büyük payı bunlara veriyordu —
# oysa fiyatları sabit kaldığı için ne kâr ne zarar ederler. Taramadan
# tamamen çıkarıyoruz.
STABLECOIN_BASES = {
    "USDT", "USDC", "DAI", "EURC", "TUSD", "USDP", "PYUSD", "GUSD",
    "FDUSD", "USDD", "USTC", "EURT", "BUSD", "LUSD", "USDS",
}

STATE_FILE = "state.json"
LOG_FILE = "trades_log.csv"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# =====================================================================


def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:
        print(f"[Telegram hata]: {e}")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"cash": STARTING_BALANCE_USD, "positions": {}, "starting_balance": STARTING_BALANCE_USD}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_trade(symbol, action, price, qty, cash_after, reason):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "symbol", "action", "price", "qty", "cash_after", "reason"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), symbol, action,
                          f"{price:.6f}", f"{qty:.6f}", f"{cash_after:.2f}", reason])


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def get_top_symbols(exchange, n):
    """Hacme göre en yüksek n USDT paritesini döndürür (tek API çağrısı)."""
    tickers = exchange.fetch_tickers()
    rows = []
    for symbol, t in tickers.items():
        if not symbol.endswith(f"/{QUOTE_CURRENCY}"):
            continue
        base = symbol.split("/")[0]
        if base in STABLECOIN_BASES:
            continue
        if t.get("quoteVolume") is None:
            continue
        rows.append((symbol, t["quoteVolume"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in rows[:n]]


def analyze_symbol(exchange, symbol):
    """Bir coin için sinyal + volatilite (risk skoru) hesaplar. Hata olursa None döner."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
        if len(ohlcv) < MA_SLOW + 2:
            return None
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
        df["ma_fast"] = df["close"].rolling(MA_FAST).mean()
        df["ma_slow"] = df["close"].rolling(MA_SLOW).mean()
        df["returns"] = df["close"].pct_change()

        latest, prev = df.iloc[-1], df.iloc[-2]
        volatility = df["returns"].tail(20).std()  # risk skoru: getiri oynaklığı (std sapma)
        if pd.isna(volatility) or volatility <= 0:
            return None

        golden_cross = prev["ma_fast"] <= prev["ma_slow"] and latest["ma_fast"] > latest["ma_slow"]
        rsi_recovering = prev["rsi"] < RSI_OVERSOLD and latest["rsi"] >= RSI_OVERSOLD
        death_cross = prev["ma_fast"] >= prev["ma_slow"] and latest["ma_fast"] < latest["ma_slow"]
        rsi_dropping = prev["rsi"] > RSI_OVERBOUGHT and latest["rsi"] <= RSI_OVERBOUGHT

        return {
            "symbol": symbol,
            "price": float(latest["close"]),
            "volatility": float(volatility),
            "buy_signal": bool(golden_cross or rsi_recovering),
            "buy_reason": "MA Golden Cross" if golden_cross else ("RSI aşırı satımdan çıkış" if rsi_recovering else None),
            "sell_signal": bool(death_cross or rsi_dropping),
            "sell_reason": "MA Death Cross" if death_cross else ("RSI aşırı alımdan düşüş" if rsi_dropping else None),
        }
    except Exception as e:
        print(f"[Uyarı] {symbol} analiz edilemedi: {e}")
        return None


def main():
    state = load_state()
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({"enableRateLimit": True})

    top_symbols = get_top_symbols(exchange, TOP_N_COINS)
    print(f"📡 {len(top_symbols)} coin taranıyor (hacme göre ilk {TOP_N_COINS})...")

    results = {}
    for symbol in top_symbols:
        analysis = analyze_symbol(exchange, symbol)
        if analysis:
            results[symbol] = analysis

    # ---- 1) Açık pozisyonlarda SAT / stop-loss / take-profit kontrolü ----
    for symbol, pos in list(state["positions"].items()):
        info = results.get(symbol)
        price = info["price"] if info else pos["entry_price"]
        change_pct = (price - pos["entry_price"]) / pos["entry_price"]

        reason = None
        if change_pct <= -STOP_LOSS_PCT:
            reason = f"Stop-loss (%{STOP_LOSS_PCT*100:.0f})"
        elif change_pct >= TAKE_PROFIT_PCT:
            reason = f"Take-profit (%{TAKE_PROFIT_PCT*100:.0f})"
        elif info and info["sell_signal"]:
            reason = info["sell_reason"]

        if reason:
            proceeds = pos["qty"] * price
            pnl = proceeds - (pos["qty"] * pos["entry_price"])
            pnl_pct = (pnl / (pos["qty"] * pos["entry_price"])) * 100
            state["cash"] += proceeds
            del state["positions"][symbol]
            log_trade(symbol, "SELL", price, pos["qty"], state["cash"], f"{reason} | PnL: {pnl:.2f} ({pnl_pct:.2f}%)")
            msg = f"🔴 SAT — {symbol} @ {price:.4f} | {reason} | PnL: {pnl:.2f} USD ({pnl_pct:.2f}%)"
            print(msg)
            send_telegram(msg)

    # ---- 2) Yeni AL adaylarını belirle ----
    open_slots = MAX_CONCURRENT_POSITIONS - len(state["positions"])
    candidates = [
        r for sym, r in results.items()
        if r["buy_signal"] and sym not in state["positions"]
    ]

    if open_slots > 0 and candidates:
        # Risk-paritesi: volatilitesi düşük olana daha çok ağırlık (1/volatilite)
        inv_vol = {c["symbol"]: 1.0 / c["volatility"] for c in candidates}
        total_inv_vol = sum(inv_vol.values())

        investable_cash = state["cash"] * (1 - CASH_RESERVE_PCT)
        max_per_coin = state["starting_balance"] * MAX_ALLOC_PER_COIN_PCT

        # En güçlü adaylardan başla (ağırlığı en yüksek = en düşük riskli), slot kadarını al
        sorted_candidates = sorted(candidates, key=lambda c: inv_vol[c["symbol"]], reverse=True)[:open_slots]
        chosen_inv_vol_sum = sum(inv_vol[c["symbol"]] for c in sorted_candidates)

        for c in sorted_candidates:
            weight = inv_vol[c["symbol"]] / chosen_inv_vol_sum
            alloc = min(investable_cash * weight, max_per_coin, state["cash"])
            if alloc < MIN_TRADE_USD:
                continue
            qty = alloc / c["price"]
            state["cash"] -= alloc
            state["positions"][c["symbol"]] = {"qty": qty, "entry_price": c["price"]}
            log_trade(c["symbol"], "BUY", c["price"], qty, state["cash"],
                      f"{c['buy_reason']} | Risk ağırlıklı bütçe: {alloc:.2f} USD (volatilite: {c['volatility']:.4f})")
            msg = f"🟢 AL — {c['symbol']} @ {c['price']:.4f} | Bütçe: {alloc:.2f} USD | Sebep: {c['buy_reason']}"
            print(msg)
            send_telegram(msg)

    # ---- 3) Özet ----
    equity = state["cash"]
    for symbol, pos in state["positions"].items():
        price = results[symbol]["price"] if symbol in results else pos["entry_price"]
        equity += pos["qty"] * price
    total_pnl_pct = ((equity - state["starting_balance"]) / state["starting_balance"]) * 100
    print(f"\n📊 Toplam değer: {equity:.2f} USD | Toplam PnL: {total_pnl_pct:.2f}% | "
          f"Açık pozisyon sayısı: {len(state['positions'])}")

    save_state(state)


if __name__ == "__main__":
    main()
