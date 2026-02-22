"""
=============================================================================
  Mr.3초 신호 감지 시스템 - 코인 전용 파이썬 코드 (업비트 기반)
  RSI + MACD + 거래량 조합 매수/매도 신호 자동 감지
=============================================================================
"""

import pyupbit
import pandas as pd
import numpy as np
import time
import datetime
import os
import logging as _logging, csv as _csv

# --------------------------------------------------------------------------
# [ 사용자 설정 영역 ]
# --------------------------------------------------------------------------
access_key = "여기에_액세스키_입력"        # 업비트 액세스 키 입력
secret_key = "여기에_시크릿키_입력"        # 업비트 시크릿 키 입력

# 거래 대상 및 기본 설정
max_scan_tickers = 50            # 스캔 종목 수 제한 (API 속도 제한 방지)
tickers        = pyupbit.get_tickers(fiat="KRW")[:max_scan_tickers]
timeframe      = 'minute60'      # 시간봉 기준
fee            = 0.0005          # 거래 수수료 0.05%

# ---------- RSI 설정 ----------
rsi_period     = 14              # RSI 계산 기간
rsi_buy_level  = 30              # RSI 매수 임계값 (30 이하 = 과매도)
rsi_sell_level = 70              # RSI 매도 임계값 (70 이상 = 과매수)

# ---------- MACD 설정 ----------
macd_fast      = 12              # MACD 단기 EMA 기간
macd_slow      = 26              # MACD 장기 EMA 기간
macd_signal    = 9               # MACD 시그널 기간

# ---------- 거래량 설정 ----------
volume_period      = 20          # 평균 거래량 계산 기간
volume_multiplier  = 1.5         # 평균 거래량 대비 배수 (1.5배 이상 = 유효)

# ---------- 터틀/ATR 설정 ----------
donchian_high_period = 10
donchian_low_period  = 20
atr_period           = 20
atr_multiplier       = 3.0

# ---------- 수익 확정 설정 ----------
profit_target_1   = 0.15         # 1차 목표 수익 15%
profit_target_2   = 0.25         # 2차 목표 수익 25%
profit_target_3   = 0.40         # 3차 목표 수익 40%
partial_sell_ratio = 0.30        # 부분 매도 비율 30%

# ---------- 리스크/포지션 설정 ----------
initial_capital  = 1_000_000     # 초기 자본금 (원)
base_order_amount = 100_000      # 기본 주문 금액 (원)
max_order_amount  = 500_000      # 최대 주문 금액 (원)
max_positions     = 10           # 최대 동시 보유 종목 수

# --------------------------------------------------------------------------
# [ 업비트 클라이언트 초기화 ]
# --------------------------------------------------------------------------
try:
    upbit        = pyupbit.Upbit(access_key, secret_key)
    krw_balance  = upbit.get_balance("KRW")
    print("✅ 업비트 연동 성공")
    print(f"💵 현재 KRW 잔고: {krw_balance:,.0f} 원")
except Exception as e:
    print(f"❌ 업비트 연동 실패: {e}")
    upbit = None

# ── 거래 로그 설정 ──────────────────────────────────────────────────────────
_LOG_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mr3seconds_bot.log")
_TRADE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mr3seconds_trades.csv")

_logger = _logging.getLogger("Mr3Seconds")
_logger.setLevel(_logging.INFO)
if not _logger.handlers:
    _fh = _logging.FileHandler(_LOG_FILE, encoding='utf-8')
    _fh.setFormatter(_logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(_fh)

if not os.path.exists(_TRADE_FILE):
    with open(_TRADE_FILE, 'w', newline='', encoding='utf-8-sig') as _f:
        _csv.writer(_f).writerow(['datetime', 'ticker', 'action', 'price', 'amount_krw', 'profit_pct', 'reason'])

def log_trade(ticker, action, price, amount_krw=0, profit_pct=None, reason=''):
    """거래 이력을 로그 파일과 CSV에 동시 기록"""
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    p_str = f"{profit_pct:+.2f}" if profit_pct is not None else "-"
    _logger.info(f"[{action}] {ticker} | 가격:{price:,.0f}원 | 금액:{amount_krw:,.0f}원 | 수익률:{p_str}% | {reason}")
    with open(_TRADE_FILE, 'a', newline='', encoding='utf-8-sig') as _f:
        _csv.writer(_f).writerow([now_str, ticker, action, price, amount_krw,
                                   f"{profit_pct:.4f}" if profit_pct is not None else '', reason])
# ────────────────────────────────────────────────────────────────────────────


# --------------------------------------------------------------------------
# [ 보조지표 계산 함수 ]
# --------------------------------------------------------------------------

def get_historical_data(ticker: str, interval: str, count: int = 200):
    """업비트 OHLCV 데이터 조회"""
    try:
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
        if df is None or df.empty:
            print(f"[{ticker}] ❌ 데이터 조회 실패")
            return None
        return df
    except Exception as e:
        print(f"[{ticker}] ❌ 데이터 오류: {e}")
        return None


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI(상대강도지수) 계산"""
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series,
              fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD, 시그널, 히스토그램 계산"""
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """전체 보조지표 계산 (RSI, MACD, 거래량, 돈키안채널, ATR)"""

    # ── RSI ──────────────────────────────────────────────────────────────
    df['rsi'] = calc_rsi(df['close'], period=rsi_period)

    # ── MACD ─────────────────────────────────────────────────────────────
    df['macd'], df['macd_signal'], df['macd_hist'] = calc_macd(
        df['close'], macd_fast, macd_slow, macd_signal
    )
    # 골든크로스: 이전 봉에서 macd < signal 이었다가 현재 봉에서 macd > signal
    df['macd_prev']   = df['macd'].shift(1)
    df['signal_prev'] = df['macd_signal'].shift(1)
    df['golden_cross'] = (df['macd'] > df['macd_signal']) & \
                         (df['macd_prev'] < df['signal_prev'])
    df['dead_cross']   = (df['macd'] < df['macd_signal']) & \
                         (df['macd_prev'] > df['signal_prev'])

    # ── 거래량 ───────────────────────────────────────────────────────────
    df['avg_volume']   = df['volume'].rolling(window=volume_period).mean().shift(1)
    df['volume_ratio'] = df['volume'] / df['avg_volume']

    # ── 돈키안 채널 ──────────────────────────────────────────────────────
    df['donchian_high'] = df['high'].rolling(window=donchian_high_period).max().shift(1)
    df['donchian_low']  = df['low'].rolling(window=donchian_low_period).min().shift(1)

    # ── ATR ──────────────────────────────────────────────────────────────
    df['tr1']        = df['high'] - df['low']
    df['tr2']        = abs(df['high'] - df['close'].shift(1))
    df['tr3']        = abs(df['low']  - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr']        = df['true_range'].rolling(window=atr_period).mean()

    return df


# --------------------------------------------------------------------------
# [ 신호 감지 함수 - 핵심 로직 ]
# --------------------------------------------------------------------------

def detect_signal(latest: pd.Series) -> dict:
    """
    Mr.3초 신호 감지 로직
    우선순위: 1순위(거래량) → 2순위(MACD) → 3순위(RSI)
    반환: {'signal': 'BUY'|'SELL'|'HOLD', 'strength': 1~3, 'reason': str}
    """
    rsi          = latest['rsi']
    golden_cross = latest['golden_cross']
    dead_cross   = latest['dead_cross']
    vol_ratio    = latest['volume_ratio']
    macd_hist    = latest['macd_hist']

    volume_ok    = (vol_ratio >= volume_multiplier)

    reasons = []
    buy_score  = 0
    sell_score = 0

    # ── 1순위: 거래량 ─────────────────────────────────────────────────────
    if volume_ok:
        reasons.append(f"거래량 급증({vol_ratio:.1f}배)")
        buy_score  += 1
        sell_score += 1   # 거래량 증가는 방향 중립 (다른 지표가 방향 결정)
    else:
        reasons.append(f"거래량 미달({vol_ratio:.1f}배)")

    # ── 2순위: MACD ───────────────────────────────────────────────────────
    if golden_cross and volume_ok:
        reasons.append("MACD 골든크로스")
        buy_score += 1
    elif dead_cross and volume_ok:
        reasons.append("MACD 데드크로스")
        sell_score += 1
    elif macd_hist > 0:
        reasons.append("MACD 양전환")
        buy_score += 0.5
    elif macd_hist < 0:
        reasons.append("MACD 음전환")
        sell_score += 0.5

    # ── 3순위: RSI ────────────────────────────────────────────────────────
    if rsi <= rsi_buy_level:
        reasons.append(f"RSI 과매도({rsi:.1f})")
        buy_score += 1
    elif rsi >= rsi_sell_level:
        reasons.append(f"RSI 과매수({rsi:.1f})")
        sell_score += 1
    else:
        reasons.append(f"RSI 중립({rsi:.1f})")

    # ── 신호 판정 ─────────────────────────────────────────────────────────
    if buy_score >= 2.5 and volume_ok:
        signal   = 'BUY'
        strength = min(int(buy_score), 3)   # 1~3 강도
    elif sell_score >= 2.5 and volume_ok:
        signal   = 'SELL'
        strength = min(int(sell_score), 3)
    else:
        signal   = 'HOLD'
        strength = 0

    return {
        'signal'   : signal,
        'strength' : strength,
        'rsi'      : rsi,
        'vol_ratio': vol_ratio,
        'macd_hist': macd_hist,
        'golden'   : golden_cross,
        'dead'     : dead_cross,
        'reason'   : ' | '.join(reasons)
    }


# --------------------------------------------------------------------------
# [ 주문 실행 함수 ]
# --------------------------------------------------------------------------

def update_order_amount() -> float:
    """총 자산 기준으로 주문 금액 동적 조절"""
    try:
        krw      = upbit.get_balance("KRW")
        balances = upbit.get_balances()
        coin_list = [f"KRW-{b['currency']}" for b in balances
                     if b['currency'] != 'KRW' and f"KRW-{b['currency']}" in tickers]
        prices = pyupbit.get_current_price(coin_list) if coin_list else {}
        total = krw
        for b in balances:
            t = f"KRW-{b['currency']}"
            if t in prices:
                total += float(b['balance']) * prices[t]
        profit_pct = (total - initial_capital) / initial_capital * 100
        if   profit_pct >= 40: multiplier = 5
        elif profit_pct >= 30: multiplier = 4
        elif profit_pct >= 20: multiplier = 3
        elif profit_pct >= 10: multiplier = 2
        else:                  multiplier = 1
        return min(base_order_amount * multiplier, max_order_amount)
    except Exception as e:
        print(f"⚠️ 주문 금액 계산 오류: {e}")
        return base_order_amount


def check_profit_targets(ticker: str, current_price: float,
                         pos: dict) -> tuple:
    """단계별 수익 확정 (1차 15% / 2차 25% / 3차 40%)"""
    if not pos['has_position']:
        return False, ""
    entry   = pos['entry_price']
    profit  = (current_price - entry) / entry * 100
    stage   = pos['profit_stage']
    MIN_KRW = 5_000

    def try_sell(ratio: str, pct_label: str, stage_num: int):
        try:
            balance = upbit.get_balance(ticker.split('-')[1])
            r       = float(ratio)
            qty     = balance * r
            val     = qty * current_price
            remain  = (balance - qty) * current_price
            if val < MIN_KRW:
                return False, f"매도금액 {val:.0f}원 미달"
            sell_qty = balance if remain < MIN_KRW else qty
            upbit.sell_market_order(ticker, sell_qty)
            log_trade(ticker, f'수익확정{stage_num}차', current_price,
                      sell_qty * current_price, profit, f"{stage_num}차 ({pct_label}% 달성)")
            pos['profit_stage'] = stage_num
            pos['amount']       = 0 if remain < MIN_KRW else balance - qty
            return True, f"{stage_num}차 수익 확정 ({pct_label}% 달성)"
        except Exception as e:
            return False, str(e)

    if profit >= profit_target_3 * 100 and stage < 3:
        return try_sell(0.50, "40", 3)
    elif profit >= profit_target_2 * 100 and stage < 2:
        return try_sell(partial_sell_ratio, "25", 2)
    elif profit >= profit_target_1 * 100 and stage < 1:
        return try_sell(partial_sell_ratio, "15", 1)
    return False, ""


# --------------------------------------------------------------------------
# [ 메인 실행 루프 ]
# --------------------------------------------------------------------------

def main():
    if not upbit:
        print("❌ 업비트 연동 실패. 종료합니다.")
        return

    # 포지션 초기화
    positions = {
        t: {
            'has_position' : False,
            'entry_price'  : 0.0,
            'stop_loss'    : 0.0,
            'amount'       : 0.0,
            'profit_stage' : 0
        } for t in tickers
    }

    # 기존 보유 포지션 복원
    print("\n🔄 기존 포지션 복원 중...")
    try:
        my_balances = upbit.get_balances()
        bal_map     = {f"KRW-{b['currency']}": b for b in my_balances}
        for t in tickers:
            if t in bal_map:
                b   = bal_map[t]
                qty = float(b['balance'])
                avg = float(b['avg_buy_price'])
                cur = pyupbit.get_current_price(t)
                if cur and qty * cur > 5_000:
                    print(f"  ✅ [{t}] 포지션 복원 (평균가: {avg:,.0f}원)")
                    df = get_historical_data(t, timeframe)
                    atr_val = 0.0
                    if df is not None:
                        df      = calculate_indicators(df)
                        atr_val = df.iloc[-1]['atr']
                    positions[t].update({
                        'has_position': True,
                        'entry_price' : avg,
                        'amount'      : qty,
                        'stop_loss'   : avg - atr_val * atr_multiplier if not pd.isna(atr_val) else 0,
                        'profit_stage': 0
                    })
    except Exception as e:
        print(f"⚠️ 포지션 복원 오류: {e}")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║         Mr.3초 신호 감지 시스템 가동 시작                    ║
║  RSI({rsi_period}) + MACD({macd_fast}/{macd_slow}/{macd_signal}) + 거래량({volume_multiplier}배)        ║
║  손절 ATR×{atr_multiplier} | 수익확정 15/25/40%              ║
╚══════════════════════════════════════════════════════════╝
""")

    while True:
        now = datetime.datetime.now()
        print(f"\n{'='*60}")
        print(f"  사이클 시작: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        for ticker in tickers:
            try:
                df = get_historical_data(ticker, timeframe)
                if df is None:
                    time.sleep(0.5)
                    continue

                df      = calculate_indicators(df)
                latest  = df.iloc[-1]
                cur     = pyupbit.get_current_price(ticker)
                if cur is None:
                    continue

                sig     = detect_signal(latest)
                pos     = positions[ticker]
                stars   = "⭐" * sig['strength'] if sig['signal'] != 'HOLD' else ""

                print(f"\n[{ticker}] 현재가: {cur:,.0f}  RSI: {sig['rsi']:.1f}  "
                      f"거래량비: {sig['vol_ratio']:.1f}x  "
                      f"{'🟢 BUY' if sig['signal']=='BUY' else '🔴 SELL' if sig['signal']=='SELL' else '⚪ HOLD'} {stars}")
                print(f"         사유: {sig['reason']}")

                # ── 미보유 상태 → 매수 신호 처리 ─────────────────────────
                if not pos['has_position']:
                    active = sum(1 for p in positions.values() if p['has_position'])
                    if active >= max_positions:
                        print(f"  ⚠️ 최대 보유 종목({max_positions}개) 도달")
                        continue

                    if sig['signal'] == 'BUY' and sig['strength'] >= 2:
                        order_amt = update_order_amount()
                        krw_bal   = upbit.get_balance("KRW")
                        if krw_bal >= order_amt:
                            print(f"  🛒 매수 실행: {order_amt:,.0f}원")
                            upbit.buy_market_order(ticker, order_amt)
                            log_trade(ticker, 'BUY', cur, order_amt, reason=sig['reason'])
                            time.sleep(2)
                            avg  = upbit.get_avg_buy_price(ticker)
                            amt  = upbit.get_balance(ticker.split('-')[1])
                            atr  = latest['atr']
                            sl   = avg - atr * atr_multiplier if not pd.isna(atr) else avg * 0.92
                            pos.update({
                                'has_position': True,
                                'entry_price' : avg,
                                'amount'      : amt,
                                'stop_loss'   : sl,
                                'profit_stage': 0
                            })
                            print(f"  ✅ 매수 완료 | 평균가: {avg:,.0f} | 손절: {sl:,.0f}")
                        else:
                            print(f"  ⚠️ 잔고 부족 ({krw_bal:,.0f} < {order_amt:,.0f})")

                # ── 보유 상태 → 수익확정 / 손절 / 포지션 유지 ─────────────
                else:
                    # 수익 확정 체크
                    sold, reason = check_profit_targets(ticker, cur, pos)
                    if sold:
                        print(f"  💎 {reason}")
                        time.sleep(2)

                    # 트레일링 스톱 업데이트
                    atr = latest['atr']
                    if not pd.isna(atr):
                        new_sl = cur - atr * atr_multiplier
                        if new_sl > pos['stop_loss']:
                            print(f"  🚀 트레일링 스톱 상향: {pos['stop_loss']:,.0f} → {new_sl:,.0f}")
                            pos['stop_loss'] = new_sl

                    # 청산 조건 체크
                    exit_signal = False
                    exit_reason = ""
                    if cur < pos['stop_loss']:
                        exit_signal, exit_reason = True, "손절(트레일링 스톱)"
                    elif cur < latest['donchian_low'] or (sig['signal'] == 'SELL' and sig['strength'] >= 2):
                        exit_signal, exit_reason = True, f"청산신호({sig['reason']})"

                    if exit_signal:
                        qty = upbit.get_balance(ticker.split('-')[1])
                        if qty and qty * cur > 5_000:
                            upbit.sell_market_order(ticker, qty)
                            pl = (cur - pos['entry_price']) / pos['entry_price'] * 100
                            print(f"  🔴 매도 완료 | 사유: {exit_reason} | 수익률: {pl:+.2f}%")
                            log_trade(ticker, 'SELL', cur, qty * cur, pl, exit_reason)
                        pos.update({'has_position': False, 'entry_price': 0.0,
                                    'stop_loss': 0.0, 'amount': 0.0, 'profit_stage': 0})
                    else:
                        pl = (cur - pos['entry_price']) / pos['entry_price'] * 100
                        print(f"  🛡️ 보유 유지 | 수익률: {pl:+.2f}% | 손절가: {pos['stop_loss']:,.0f}")

                time.sleep(0.3)

            except Exception as e:
                print(f"💥 [{ticker}] 오류: {e}")
                continue

        print(f"\n⏳ 다음 사이클 대기 중 (10초)...")
        time.sleep(10)


if __name__ == "__main__":
    main()
