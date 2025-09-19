import websocket
import json
import pandas as pd
import pandas_ta as ta

# --- 配置参数 ---
symbol = "BTC-USDT"
timeframe = "1m"  # K线周期 (e.g., 1m, 5m, 1H, 1D)
rsi_period = 14
boll_period = 20
kdj_period = 9  # KDJ通常使用(9, 3, 3)作为参数

# --- 全局变量 ---
# 创建一个DataFrame来存储K线数据
columns = ["timestamp", "open", "high", "low", "close", "volume"]
klines = pd.DataFrame(columns=columns)


def calculate_indicators(df):
    """使用传入的DataFrame计算并打印技术指标"""
    if df.empty or len(df) < max(rsi_period, boll_period, kdj_period):
        print("K线数据不足，无法计算指标...")
        return

    # 确保收盘价是数值类型
    df["close"] = pd.to_numeric(df["close"])

    # 计算RSI
    rsi = df.ta.rsi(length=rsi_period)

    # 计算布林带
    boll = df.ta.bbands(length=boll_period)

    # 计算KDJ
    kdj = df.ta.kdj(length=kdj_period)

    # 合并指标到主DataFrame
    df_with_indicators = pd.concat([df, rsi, boll, kdj], axis=1)

    # 获取并打印最新的指标
    latest_data = df_with_indicators.iloc[-1]

    print("\n--- 最新技术指标 ---")
    print(f"K线时间: {pd.to_datetime(latest_data['timestamp'], unit='ms')}")
    print(f"收盘价: {latest_data['close']:.2f}")
    print(f"RSI({rsi_period}): {latest_data[f'RSI_{rsi_period}']:.2f}")
    print(f"布林带({boll_period}):")
    print(f"  - 上轨 (BBU): {latest_data[f'BBU_{boll_period}_2.0']:.2f}")
    print(f"  - 中轨 (BBM): {latest_data[f'BBM_{boll_period}_2.0']:.2f}")
    print(f"  - 下轨 (BBL): {latest_data[f'BBL_{boll_period}_2.0']:.2f}")
    print(f"KDJ({kdj_period},3,3):")
    print(f"  - K: {latest_data[f'K_{kdj_period}_3']:.2f}")
    print(f"  - D: {latest_data[f'D_{kdj_period}_3']:.2f}")
    print(f"  - J: {latest_data[f'J_{kdj_period}_3']:.2f}")
    print("--------------------")


def on_message(ws, message):
    """处理从WebSocket接收到的消息"""
    global klines
    data = json.loads(message)

    # 检查是否是订阅成功的消息
    if "event" in data and data["event"] == "subscribe":
        print(f"成功订阅频道: {data['arg']['channel']}")
        return

    # 检查是否是心跳或错误消息
    if "event" in data or "data" not in data:
        return

    arg = data.get("arg", {})
    channel = arg.get("channel", "")
    # --- 处理实时价格数据 ---
    if channel == "tickers":
        ticker_data = data["data"][0]
        print(f"\r实时价格 ({symbol}): {ticker_data['last']}", end="", flush=True)

    # --- 处理K线数据 ---
    elif channel == f"candle{timeframe}":
        kline_data = data["data"][0]
        timestamp = int(kline_data[0])

        # 检查是否是新的K线
        if not klines.empty and klines["timestamp"].iloc[-1] == timestamp:
            # 更新当前这根未闭合的K线
            klines.iloc[-1] = kline_data
        else:
            # 添加一根新的K线
            new_row = pd.DataFrame([kline_data], columns=columns)
            klines = pd.concat([klines, new_row], ignore_index=True)
            print(
                f"\n收到新的 {timeframe} K线 - 时间: {pd.to_datetime(timestamp, unit='ms')}"
            )

        # 当有足够数据时，重新计算指标
        calculate_indicators(klines)


def on_error(ws, error):
    """处理错误"""
    print(f"发生错误: {error}")


def on_close(ws, close_status_code, close_msg):
    """处理连接关闭"""
    print("### 连接已关闭 ###")


def on_open(ws):
    """当WebSocket连接建立时调用"""
    print("### 连接已建立 ###")

    # 构建订阅消息
    subscribe_message = {
        "op": "subscribe",
        "args": [
            {"channel": "tickers", "instId": symbol},
            {"channel": f"candle{timeframe}", "instId": symbol},
        ],
    }
    # 发送订阅请求
    ws.send(json.dumps(subscribe_message))


if __name__ == "__main__":
    # OKX WebSocket 公共频道地址
    ws_url = "wss://ws.okx.com:8443/ws/v5/public"

    # 开启WebSocket连接
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    # 运行WebSocket客户端
    ws.run_forever()
