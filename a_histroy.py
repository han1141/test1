import akshare as ak
import pandas as pd
import datetime
import time


def get_stock_hist_data(symbol, period="daily", start_date=None, end_date=None):
    """
    使用akshare获取A股历史数据。
    
    Args:
        symbol: 股票代码，如 "000001"
        period: 数据周期，"daily"为日线数据
        start_date: 开始日期，格式为 "20240101"
        end_date: 结束日期，格式为 "20240131"
    """
    print(f"开始从akshare获取股票 {symbol} 的历史数据...")
    
    try:
        # 使用akshare获取A股历史数据
        # period参数: "daily"=日K, "weekly"=周K, "monthly"=月K
        # adjust参数: ""=不复权, "qfq"=前复权, "hfq"=后复权
        # 对于分钟级数据，使用正确的函数和参数
        if period in ["1", "5", "15", "30", "60"]:
            # 分钟级数据
            stock_data = ak.stock_zh_a_minute(
                symbol=symbol,
                period=period,
                adjust="qfq"
            )
        else:
            # 日级数据
            stock_data = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date.split()[0].replace("-", ""),  # 转换为YYYYMMDD格式
                end_date=end_date.split()[0].replace("-", ""),
                adjust="qfq"
            )
        
        if stock_data is None or stock_data.empty:
            print(f"未能获取到股票 {symbol} 的数据")
            return None
            
        print(f"成功获取到 {len(stock_data)} 条历史数据")
        return stock_data
        
    except Exception as e:
        print(f"获取股票数据时发生错误: {e}")
        return None


def main():
    """主函数，执行数据获取、处理和保存"""
    # 1. 设置参数
    symbol = "601138"  # 平安银行，可以修改为其他A股代码
    period = "daily"   # 60分钟K线数据

    # 2. 计算时间范围（最近22天）
    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=1000)

    # 对于分钟级数据，akshare需要的日期格式为 "YYYY-MM-DD HH:MM:SS"
    # 对于日级数据，需要 "YYYYMMDD" 格式
    if period in ["1", "5", "15", "30", "60"]:
        # 分钟级数据使用完整时间格式
        start_date = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_date = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        # 日级数据使用简单日期格式
        start_date = start_dt.strftime("%Y%m%d")
        end_date = end_dt.strftime("%Y%m%d")

    data_type = "分钟K线" if period in ["1", "5", "15", "30", "60"] else "日K线"
    print(f"计划获取股票 {symbol} 从 {start_dt.date()} 到 {end_dt.date()} 的{period}{data_type}数据...")

    # 3. 获取股票历史数据
    stock_data = get_stock_hist_data(symbol, period, start_date, end_date)

    if stock_data is None or stock_data.empty:
        print("未能获取到任何数据，程序退出。")
        return

    # 4. 处理数据格式以匹配原有的CSV格式
    # 先打印原始数据的列名，以便调试
    print(f"原始数据列名: {list(stock_data.columns)}")
    print(f"原始数据前5行:")
    print(stock_data.head())
    
    df_final = stock_data.copy()
    
    # 重命名列以匹配原有格式 - 根据实际的akshare返回列名调整
    column_mapping = {
        '时间': 'timestamps',
        '日期': 'timestamps',
        'datetime': 'timestamps',
        'time': 'timestamps',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume',
        '成交额': 'amount'
    }
    
    # 只保留需要的列并重命名
    available_cols = [col for col in column_mapping.keys() if col in df_final.columns]
    print(f"可用的列: {available_cols}")
    
    if available_cols:
        df_final = df_final[available_cols].rename(columns=column_mapping)
    else:
        # 如果没有找到匹配的列名，尝试使用索引
        print("未找到匹配的列名，尝试使用默认列名...")
        if len(df_final.columns) >= 6:
            df_final.columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume'] + list(df_final.columns[6:])
    
    # 确保timestamps列为datetime格式
    if 'timestamps' in df_final.columns:
        df_final['timestamps'] = pd.to_datetime(df_final['timestamps'])
    
    # 确保数值列为数值类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    # 按时间排序
    if 'timestamps' in df_final.columns:
        df_final = df_final.sort_values('timestamps').reset_index(drop=True)

    # 5. 保存为CSV文件
    output_filename = f"{symbol}_{period}min_history.csv" if period in ["1", "5", "15", "30", "60"] else f"{symbol}_daily_history.csv"
    df_final.to_csv(output_filename, index=False)

    print(f"\n数据处理完成！")
    data_type = f"{period}分钟K线" if period in ["1", "5", "15", "30", "60"] else "日K线"
    print(f"总共获取了 {len(df_final)} 条{data_type}记录。")
    print(f"数据已成功保存到文件: {output_filename}")
    print(f"数据列: {list(df_final.columns)}")


if __name__ == "__main__":
    main()
