import os
import pandas as pd
import akshare as ak
from dotenv import load_dotenv
from typing import Dict, List, Any
from datetime import date, timedelta
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage

# 本地
from my_feedparser import get_strategy_reports

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

now_time = date.today()
prev_time = now_time - timedelta(days=30)
print(prev_time)


class StockAnalysisState(TypedDict):
    message: Annotated[List, add_messages]


def llm(model="deepseek-chat", temperature=0.1):
    return ChatOpenAI(
        model=model,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
    )


def get_all_a_stock_realtime():
    """获取所有A股的实时行情数据"""
    print("正在获取A股实时行情数据...")
    try:
        large_cap_stocks_df = ak.stock_zh_a_spot_em()
        # 数据清洗：将字符串格式的数值转换为浮点数，并处理异常值
        cols_to_numeric = [
            "最新价",
            "涨跌幅",
            "市盈率-动态",
            "市净率",
            "总市值",
            "换手率",
        ]
        stock_df = large_cap_stocks_df[
            ~large_cap_stocks_df["代码"].str.startswith("688")
        ]
        for col in cols_to_numeric:
            stock_df[col] = pd.to_numeric(stock_df[col], errors="coerce")
        stock_df.dropna(subset=cols_to_numeric, inplace=True)
        print(f"成功获取并处理了 {len(stock_df)} 只A股的实时数据。")
        return stock_df
    except Exception as e:
        print(f"获取AKShare数据失败: {e}")
        return pd.DataFrame()


def analyze_reports_and_select_stocks(reports, stock_df):
    if len(reports) == 0:
        print("未能获取到任何报告，请检查RSS链接或网络。")
        return
    """
    分析报告并根据策略筛选股票
    """
    # AI策略核心：定义我们关注的正面主题/行业关键词
    # 在更复杂的AI模型中，这些关键词可以通过NLP技术自动提取和加权
    positive_keywords = [
        # --- 1. TMT (科技、媒体和通信) ---
        # a. 人工智能 & 计算
        "人工智能",
        "AI",
        "AIGC",
        "多模态",
        "大模型",
        "算力",
        "数据中心",
        "液冷",
        "边缘计算",
        "云计算",
        "量子计算",
        "量子通信",
        # b. 半导体 & 芯片
        "半导体",
        "芯片",
        "集成电路",
        "EDA",
        "光刻机",
        "Chiplet",
        "第三代半导体",
        "氮化镓",
        "碳化硅",
        "IGBT",
        "封测",
        # c. 软件 & 信息技术
        "信创",
        "国产软件",
        "操作系统",
        "数据库",
        "ERP",
        "工业软件",
        "网络安全",
        "鸿蒙",
        "欧拉",
        "SaaS",
        # d. 消费电子 & 物联网
        "消费电子",
        "智能手机",
        "VR",
        "AR",
        "MR",
        "元宇宙",
        "可穿戴设备",
        "智能家居",
        "物联网",
        "传感器",
        "卫星互联网",
        # --- 2. 高端制造 & 工业 ---
        "高端制造",
        "工业母机",
        "数控机床",
        "机器人",
        "工业机器人",
        "服务机器人",
        "自动化",
        "智能制造",
        "工业4.0",
        "3D打印",
        "专精特新",
        # --- 3. 新能源 & 汽车 ---
        # a. 能源生产与存储
        "新能源",
        "光伏",
        "HJT",
        "TOPCon",
        "钙钛矿",
        "风电",
        "海风",
        "储能",
        "锂电池",
        "钠电池",
        "固态电池",
        "氢能源",
        "燃料电池",
        "特高压",
        "虚拟电厂",
        # b. 智能汽车产业链
        "汽车",
        "智能驾驶",
        "无人驾驶",
        "新能源汽车",
        "汽车电子",
        "激光雷达",
        "毫米波雷达",
        "一体化压铸",
        "充电桩",
        "换电",
        "飞行汽车",
        "低空经济",
        # --- 4. 生物医药 & 大健康 ---
        "生物医药",
        "创新药",
        "CXO",
        "CRO",
        "CDMO",
        "细胞治疗",
        "基因编辑",
        "ADC药物",
        "mRNA",
        "医疗器械",
        "高端医疗设备",
        "体外诊断",
        "脑机接口",
        "合成生物学",
        # --- 5. 新材料 ---
        "新材料",
        "碳纤维",
        "复合材料",
        "特种钢材",
        "高温合金",
        "稀土",
        "永磁材料",
        "光刻胶",
        "电子特气",
        "OLED材料",
        "气凝胶",
        # --- 6. 航空航天 & 国防军工 ---
        "航空航天",
        "大飞机",
        "商业航天",
        "卫星",
        "火箭",
        "国防军工",
        "无人机",
        # --- 7. 大消费 (细分领域) ---
        "国货潮牌",
        "新消费",
        "预制菜",
        "跨境电商",
        "免税",
        "旅游",
        "酒店",
        "医美",
        # --- 8. 其他前沿科技 ---
        "人形机器人",
        "生物制造",
        "绿色经济",
        "循环经济",
    ]

    print(f"\n策略关键词：{', '.join(positive_keywords)}")

    # 从最新报告中寻找包含关键词的报告
    triggered_reports = []
    for report in reports:
        for keyword in positive_keywords:
            if keyword in report["description"]:
                triggered_reports.append((keyword, report["title"]))
                break  # 一篇报告匹配一个关键词即可

    if not triggered_reports:
        print("\n在最新的报告中未发现明确的策略主题，暂不进行选股。")
        return pd.DataFrame()

    print("\n发现以下策略主题报告：")
    for keyword, title in triggered_reports:
        print(f"- 主题'{keyword}': {title}")

    # 定义量化筛选规则
    # 这是一个示例规则，你可以根据自己的策略进行调整
    # 规则1：市盈率(PE)在0到50之间 (剔除亏损和估值过高的)
    # 规则2：总市值大于200亿 (选择龙头或中盘股)
    # 规则3：当日涨跌幅小于5% (避免追高)
    print("\n执行量化筛选，规则如下：")
    print("1. 市盈率(PE) > 0 and 市盈率(PE) < 50")
    print("2. 总市值 > 200亿")
    print("3. 涨跌幅 < 5%")

    filtered_stocks = stock_df[
        (stock_df["市盈率-动态"] > 0)
        & (stock_df["市盈率-动态"] < 50)
        & (stock_df["总市值"] > 200 * 10**8)
        & (stock_df["涨跌幅"] < 5)
    ].copy()

    # 根据换手率进行排序，选择市场关注度较高的
    final_selection = filtered_stocks.sort_values(by="换手率", ascending=False)

    return final_selection


if __name__ == "__main__":
    reports_data = get_strategy_reports()
    stock_data_df = get_all_a_stock_realtime()

    if not reports_data or stock_data_df.empty:
        print("\n数据获取失败，程序终止。")
    else:
        # 2. 执行AI选股策略
        selected_stocks = analyze_reports_and_select_stocks(reports_data, stock_data_df)

        # 3. 输出结果
        if not selected_stocks.empty:
            print("\n===================== AI选股结果 =====================")
            # 为了界面整洁，只选择部分关键列进行展示
            display_columns = [
                "代码",
                "名称",
                "最新价",
                "涨跌幅",
                "换手率",
                "市盈率-动态",
                "总市值",
            ]
            print(selected_stocks[display_columns])  # 展示前50名
            print("======================================================")
            # --- 新增功能：将选股结果导出为CSV文件 ---
            try:
                # 定义要导出的文件名
                output_filename = "ai_selected_stocks.csv"

                # 调用 to_csv 方法，将完整的 selected_stocks DataFrame 保存到文件
                # index=False 表示不将DataFrame的索引写入文件
                # encoding='utf-8-sig' เพื่อป้องกันการแสดงผลภาษาจีนเป็นตัวอักษรที่อ่านไม่ออกใน Excel
                selected_stocks.to_csv(
                    output_filename, index=False, encoding="utf-8-sig"
                )

                print(f"\n选股结果已成功导出到文件: {output_filename}")
            except Exception as e:
                print(f"\n导出文件时发生错误: {e}")
            # ---------------------------------------------
            print("\n*注意：以上结果仅为基于预设策略的演示，不构成任何投资建议。")
        else:
            print("\n根据当前策略，未筛选出符合条件的股票。")
