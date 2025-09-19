# -*- coding: utf-8 -*-

"""
一个实时的技术指标计算脚本。
功能：
1. 启动时，获取近期的K线数据作为计算基础。
2. 连接到 Finnhub WebSocket 并订阅实时成交价。
3. 每当有新价格时，更新K线数据。
4. 实时重新计算 KDJ 和 RSI 指标并打印最新的值。
"""

import websocket
import json
import pandas as pd
import pandas_ta as ta
import finnhub
from datetime import datetime, timedelta

# --- 全局变量，用于存储我们的K线数据 ---
# 我们将使用1分钟K线作为例子
hist_data = None


# --- 核心: 计算并打印指标的函数 ---
def calculate_and_print_indicators(data_df: pd.DataFrame):
    """
    在此函数中，我们使用 pandas-ta 来计算指标。
    """
    if data_df is None or len(data_df) < 20:  # 确保有足够的数据
        print("数据量不足，无法计算指标...")
        return

    # 1. 复制一份DataFrame，以防计算过程弄乱原始数据
    df = data_df.copy()

    # 2. 使用 pandas-ta 计算 KDJ 和 RSI
    # .ta 会自动在DataFrame后面附加新的列，例如 'K_9_3', 'D_9_3', 'RSI_14'
    df.ta.kdj(append=True)
    df.ta.rsi(append=True)

    # 3. 清理掉因为计算初期产生的NaN值
    df.dropna(inplace=True)

    # 4. 获取并打印最新的指标值
    latest_indicators = df.iloc[-1]  # 获取最后一行数据

    latest_price = latest_indicators["Close"]
    latest_rsi = latest_indicators["RSI_14"]
    latest_k = latest_indicators["K_9_3"]
    latest_d = latest_indicators["D_9_3"]

    # 使用 \r 实现单行刷新，让输出更整洁
    print(
        f"\r最新价格: {latest_price:8.2f} | "
        f"RSI(14): {latest_rsi:5.2f} | "
        f"K: {latest_k:5.2f} | "
        f"D: {latest_d:5.2f}",
        end="",  # end="" 防止打印后换行
    )


# --- WebSocket 事件处理函数 ---


def on_message(ws, message):
    """当接收到WebSocket消息时被调用"""
    global hist_data

    try:
        data = json.loads(message)
        if data["type"] == "trade":
            for trade in data["data"]:
                trade_price = trade["p"]

                # --- 关键逻辑: 更新我们的数据集 ---
                # 获取当前时间的分钟数，用于确定更新哪一根K线
                current_minute = datetime.fromtimestamp(trade["t"] / 1000).replace(
                    second=0, microsecond=0
                )
                last_known_minute = hist_data.index[-1]

                if current_minute == last_known_minute:
                    # 如果还是在当前K线的时间内，则更新
                    hist_data.loc[last_known_minute, "Close"] = trade_price
                    if trade_price > hist_data.loc[last_known_minute, "High"]:
                        hist_data.loc[last_known_minute, "High"] = trade_price
                    if trade_price < hist_data.loc[last_known_minute, "Low"]:
                        hist_data.loc[last_known_minute, "Low"] = trade_price

                elif current_minute > last_known_minute:
                    # 如果进入了新的K线（新的一分钟）
                    # 创建一个新的K线行
                    new_candle = {
                        "Open": trade_price,
                        "High": trade_price,
                        "Low": trade_price,
                        "Close": trade_price,
                    }
                    # 添加到DataFrame中，并移除最老的一根，保持数据量稳定
                    hist_data = pd.concat(
                        [
                            hist_data.iloc[1:],
                            pd.DataFrame(new_candle, index=[current_minute]),
                        ]
                    )

                # --- 数据更新后，立即重新计算指标 ---
                calculate_and_print_indicators(hist_data)

    except Exception as e:
        # 简单打印错误，实际应用中应做得更完善
        print(f"\n处理消息时出错: {e}")


def on_error(ws, error):
    print(f"WebSocket 错误: {error}")


def on_close(ws, close_status_code, close_msg):
    print("\n### WebSocket 连接已关闭 ###")


def on_open(ws):
    print("### WebSocket 连接成功，正在订阅... ###")
    ws.send('{"type":"subscribe","symbol":"BINANCE:BTCUSDT"}')


# --- 主执行模块 ---
if __name__ == "__main__":

    FINNHUB_API_KEY = "d2rr0m1r01qv11lfs060d2rr0m1r01qv11lfs06g"  # <--- 在这里填入您的API密钥

    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "你的_API_密钥_放在这里":
        print("错误：请在代码中设置您的 FINNHUB_API_KEY。")
        exit()

    # --- 步骤1: 获取初始K线数据作为计算基础 ---
    print("正在获取初始K线数据...")
    try:
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        # 获取过去300分钟的1分钟K线数据
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(minutes=300)).timestamp())

        candles = finnhub_client.stock_candles(
            "BINANCE:BTCUSDT", "1", start_time, end_time
        )

        if candles["s"] == "ok" and len(candles["c"]) > 0:
            hist_data = pd.DataFrame(
                {
                    "Open": candles["o"],
                    "High": candles["h"],
                    "Low": candles["l"],
                    "Close": candles["c"],
                }
            )
            timestamps = [
                datetime.fromtimestamp(ts).replace(second=0, microsecond=0)
                for ts in candles["t"]
            ]
            hist_data.index = pd.DatetimeIndex(timestamps)
            print("初始数据获取成功！")
        else:
            print("无法获取初始数据，程序退出。")
            exit()
    except Exception as e:
        print(f"获取初始数据时出错: {e}, 程序退出。")
        exit()

    # --- 步骤2: 启动 WebSocket ---
    # websocket.enableTrace(True)  # 如果需要详细调试信息，可以取消本行注释
    ws_url = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever()
