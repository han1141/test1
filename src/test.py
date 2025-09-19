import os
import pandas as pd
import akshare as ak
from datetime import datetime
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-a6693eeca99847a8b70a5e7afdbac906"  # 请替换为您的DeepSeek API密钥
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

class StockAnalysisState(TypedDict):
    messages: Annotated[List, add_messages]
    stock_data: Annotated[pd.DataFrame, lambda x, y: y if not y.empty else x]
    news_data: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    final_recommendations: List[Dict[str, Any]]

def get_stock_data():
    """获取A股实时行情数据"""
    try:
        # 获取A股实时行情
        stock_zh_a_spot_df = ak.stock_zh_a_spot_em()
        non_kcb_stocks_df = stock_zh_a_spot_df[~stock_zh_a_spot_df['代码'].str.startswith('688')]
        return non_kcb_stocks_df
    except Exception as e:
        print(f"获取股票数据失败: {e}")
        return None

def get_detailed_stock_info(stock_codes: List[str]) -> pd.DataFrame:
    """获取股票详细信息"""
    detailed_data = []
    for code in stock_codes[:10]:  # 限制数量避免API限制
        try:
            # 获取股票历史数据
            hist_data = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20250101", adjust="")
            if not hist_data.empty:
                latest = hist_data.iloc[-1]
                stock_info = {
                    '代码': code,
                    '收盘价': latest['收盘'],
                    '成交量': latest['成交量'],
                    '成交额': latest['成交额'],
                    '振幅': latest['振幅'],
                    '涨跌幅': latest['涨跌幅'],
                    '涨跌额': latest['涨跌额'],
                    '换手率': latest['换手率']
                }
                detailed_data.append(stock_info)
        except Exception as e:
            print(f"获取股票 {code} 详细信息失败: {e}")
    
    return pd.DataFrame(detailed_data)

def get_stock_news():
    """获取股市新闻"""
    try:
        # 获取股票新闻
        news_df = ak.stock_news_em()
        return news_df.head(20)  # 获取最新10条新闻
    except Exception as e:
        print(f"获取新闻数据失败: {e}")
        return None

def get_industry_news():
    """获取行业资讯"""
    try:
        # 获取宏观经济数据相关新闻
        macro_news = ak.news_cctv()
        return macro_news.head(10)  # 获取最新5条宏观新闻
    except Exception as e:
        print(f"获取行业新闻失败: {e}")
        return None

def data_collection_node(state: StockAnalysisState) -> StockAnalysisState:
    """数据收集节点"""
    print("正在收集股票数据...")
    stock_data = get_stock_data()
    
    if stock_data is not None:
        # 基础筛选
        filtered_stocks = stock_data[
            (stock_data['涨跌幅'] > -5) &  # 跌幅不超过5%
            (stock_data['成交量'] > stock_data['成交量'].median()) &
            (stock_data['流通市值'] > 1000000000)
        ].head(100)
        
        state["stock_data"] = filtered_stocks
        state["messages"].append(HumanMessage(content=f"已收集到 {len(filtered_stocks)} 只股票的基础数据"))
    
    return state

def news_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    """新闻分析节点"""
    print("正在收集和分析新闻...")
    
    # 收集新闻数据
    stock_news = get_stock_news()
    industry_news = get_industry_news()
    
    news_content = ""
    
    if stock_news is not None and not stock_news.empty:
        print(f"✅ 获取到 {len(stock_news)} 条股票新闻")
        news_content += "=== 股票市场新闻 ===\n"
        for _, row in stock_news.iterrows():
            title = row.get('标题', '无标题')
            time = row.get('发布时间', '时间未知')
            content = row.get('内容', '无内容')[:200]  # 限制长度
            news_content += f"标题: {title}\n时间: {time}\n内容: {content}...\n\n"
    
    if industry_news is not None and not industry_news.empty:
        print(f"✅ 获取到 {len(industry_news)} 条行业新闻")
        news_content += "\n=== 宏观经济新闻 ===\n"
        for _, row in industry_news.iterrows():
            title = row.get('标题', '无标题')
            time = row.get('时间', '时间未知')
            news_content += f"标题: {title}\n时间: {time}\n\n"
    
    # 使用AI分析新闻
    if news_content:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.1
        )
        
        news_prompt = f"""
        请分析以下股票市场和宏观经济新闻，提取对股票投资有影响的关键信息：
        
        {news_content}
        
        请从以下角度分析：
        1. 市场整体情绪和趋势
        2. 政策影响和监管变化
        3. 行业热点和机会
        4. 风险因素和警示
        5. 对个股选择的影响
        
        请以JSON格式返回分析结果，包含市场情绪评分(1-10分)和关键要点。
        """
        
        try:
            print("🔄 正在调用DeepSeek进行新闻分析...")
            response = llm.invoke([
                SystemMessage(content="你是一个专业的财经新闻分析师，擅长从新闻中提取投资价值信息。"),
                HumanMessage(content=news_prompt)
            ])
            
            print("✅ DeepSeek新闻分析调用成功")
            state["news_data"] = {
                "raw_news": news_content,
                "analysis": response.content
            }
        except Exception as e:
            print(f"❌ DeepSeek新闻分析调用失败: {e}")
            state["news_data"] = {
                "raw_news": news_content,
                "error": str(e)
            }
    else:
        print("⚠️ 未获取到新闻数据")
        state["news_data"] = {"error": "无新闻数据"}
    
    state["messages"].append(HumanMessage(content="新闻分析完成"))
    return state

def test_deepseek_connection():
    """测试DeepSeek连接"""
    print("正在测试DeepSeek API连接...")
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.1
        )
        
        # 发送测试消息
        response = llm.invoke([HumanMessage(content="测试连接，请回复'连接成功'")])
        print("✅ DeepSeek API连接成功")
        return True
    except Exception as e:
        print(f"❌ DeepSeek API连接失败: {e}")
        return False

def technical_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    """技术分析节点"""
    print("正在进行技术分析...")
    
    # 初始化DeepSeek模型
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.1
    )
    
    stock_data = state["stock_data"]
    if stock_data is not None and not stock_data.empty:
        # 准备技术分析数据
        top_stocks = stock_data.head(30)
        stock_summary = top_stocks[['代码', '名称', '最新价', '涨跌幅', '成交量', '流通市值']].to_string()
        
        technical_prompt = f"""
        作为专业的股票技术分析师，请分析以下股票数据：
        
        {stock_summary}
        
        请从以下角度进行技术分析：
        1. 价格趋势分析
        2. 成交量分析
        3. 市值规模评估
        4. 短期技术指标判断
        5. 风险评估
        
        请以JSON格式返回分析结果，包含每只股票的技术评分(1-10分)和分析要点。
        """
        
        try:
            print("🔄 正在调用DeepSeek进行技术分析...")
            response = llm.invoke([
                SystemMessage(content="你是一个专业的股票技术分析师，擅长技术指标分析和市场趋势判断。"),
                HumanMessage(content=technical_prompt)
            ])
            
            print("✅ DeepSeek技术分析调用成功")
            state["technical_analysis"] = {"analysis": response.content}
            state["messages"].append(HumanMessage(content="技术分析完成"))
        except Exception as e:
            print(f"❌ DeepSeek技术分析调用失败: {e}")
            state["technical_analysis"] = {"error": str(e)}
    
    return state

def fundamental_analysis_node(state: StockAnalysisState) -> StockAnalysisState:
    """基本面分析节点"""
    print("正在进行基本面分析...")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.1
    )
    
    stock_data = state["stock_data"]
    if stock_data is not None and not stock_data.empty:
        top_stocks = stock_data.head(30)
        
        # 检查可用列并选择存在的列
        available_cols = ['代码', '名称', '流通市值']
        optional_cols = ['市盈率', '市净率', '总市值', '市销率']
        
        # 添加存在的可选列
        for col in optional_cols:
            if col in top_stocks.columns:
                available_cols.append(col)
        
        # 如果没有估值指标，使用基础指标
        if len(available_cols) == 3:  # 只有基础三列
            available_cols.extend(['最新价', '涨跌幅', '成交量'])
        
        # 过滤出实际存在的列
        display_cols = [col for col in available_cols if col in top_stocks.columns]
        
        fundamental_prompt = f"""
        请对以下股票进行基本面分析：
        
        股票列表：
        {top_stocks[display_cols].to_string()}
        
        请分析：
        1. 估值水平分析（基于可用数据）
        2. 市值规模合理性  
        3. 行业地位和前景
        4. 财务健康度评估
        5. 投资价值评级
        
        请以JSON格式返回，包含每只股票的基本面评分(1-10分)和核心观点。
        """
        
        try:
            print("🔄 正在调用DeepSeek进行基本面分析...")
            response = llm.invoke([
                SystemMessage(content="你是一个专业的股票基本面分析师，精通财务分析和估值模型。"),
                HumanMessage(content=fundamental_prompt)
            ])
            
            print("✅ DeepSeek基本面分析调用成功")
            state["fundamental_analysis"] = {"analysis": response.content}
            state["messages"].append(HumanMessage(content="基本面分析完成"))
        except Exception as e:
            print(f"❌ DeepSeek基本面分析调用失败: {e}")
            state["fundamental_analysis"] = {"error": str(e)}
    
    return state

def final_recommendation_node(state: StockAnalysisState) -> StockAnalysisState:
    """最终推荐节点"""
    print("正在生成最终投资建议...")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.2
    )
    
    # 整合所有分析结果
    technical = state.get("technical_analysis", {})
    fundamental = state.get("fundamental_analysis", {})
    news = state.get("news_data", {})
    
    final_prompt = f"""
    基于以下技术分析、基本面分析和新闻分析结果，请给出最终的投资建议：
    
    技术分析结果：
    {technical.get('analysis', '暂无技术分析')}
    
    基本面分析结果：
    {fundamental.get('analysis', '暂无基本面分析')}
    
    新闻分析结果：
    {news.get('analysis', '暂无新闻分析')}
    
    请提供：
    1. 推荐股票排序（按投资价值）
    2. 每只股票的买入建议（强烈推荐/推荐/观望/回避）
    3. 建议仓位配置
    4. 风险提示（结合新闻情绪）
    5. 预期收益评估
    6. 市场时机判断（基于新闻分析）
    
    请以结构化的JSON格式返回最终建议。
    """
    
    try:
        print("🔄 正在调用DeepSeek生成最终投资建议...")
        response = llm.invoke([
            SystemMessage(content="你是一个资深的投资顾问，能够综合技术分析和基本面分析给出专业的投资建议。"),
            HumanMessage(content=final_prompt)
        ])
        
        print("✅ DeepSeek最终建议生成成功")
        state["final_recommendations"] = {"recommendations": response.content}
        state["messages"].append(HumanMessage(content="最终投资建议生成完成"))
    except Exception as e:
        print(f"❌ DeepSeek最终建议生成失败: {e}")
        state["final_recommendations"] = {"error": str(e)}
    
    return state

def create_stock_analysis_workflow():
    """创建股票分析工作流"""
    # 创建工作流图
    workflow = StateGraph(StockAnalysisState)
    
    # 添加节点
    workflow.add_node("data_collection", data_collection_node)
    workflow.add_node("news_analysis", news_analysis_node)
    workflow.add_node("technical_analysis", technical_analysis_node)
    workflow.add_node("fundamental_analysis", fundamental_analysis_node)
    workflow.add_node("final_recommendation", final_recommendation_node)
    
    # 设置串行工作流程，避免并发更新
    workflow.set_entry_point("data_collection")
    workflow.add_edge("data_collection", "news_analysis")
    workflow.add_edge("news_analysis", "technical_analysis")
    workflow.add_edge("technical_analysis", "fundamental_analysis")
    workflow.add_edge("fundamental_analysis", "final_recommendation")
    workflow.add_edge("final_recommendation", END)
    
    return workflow.compile()

def save_analysis_to_excel(state: StockAnalysisState):
    """保存分析结果到Excel"""
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ai_stock_analysis_{current_date}.xlsx"
    filepath = os.path.join(os.getcwd(), filename)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # 保存原始股票数据
        if not state["stock_data"].empty:
            state["stock_data"].to_excel(writer, sheet_name='原始数据', index=False)
        
        # 保存分析结果
        analysis_summary = []
        
        # 技术分析结果
        if "technical_analysis" in state and "analysis" in state["technical_analysis"]:
            analysis_summary.append({
                "分析类型": "技术分析",
                "结果": state["technical_analysis"]["analysis"]
            })
        
        # 基本面分析结果
        if "fundamental_analysis" in state and "analysis" in state["fundamental_analysis"]:
            analysis_summary.append({
                "分析类型": "基本面分析", 
                "结果": state["fundamental_analysis"]["analysis"]
            })
        
        # 新闻分析结果
        if "news_data" in state and "analysis" in state["news_data"]:
            analysis_summary.append({
                "分析类型": "新闻分析",
                "结果": state["news_data"]["analysis"]
            })
        
        # 最终建议
        if "final_recommendations" in state and "recommendations" in state["final_recommendations"]:
            analysis_summary.append({
                "分析类型": "最终投资建议",
                "结果": state["final_recommendations"]["recommendations"]
            })
        
        if analysis_summary:
            pd.DataFrame(analysis_summary).to_excel(writer, sheet_name='AI分析结果', index=False)
    
    print(f"AI分析结果已保存到: {filepath}")
    return filepath

def generate_markdown_report(state: StockAnalysisState):
    """生成Markdown分析报告"""
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ai_stock_analysis_report_{current_date}.md"
    filepath = os.path.join(os.getcwd(), filename)
    
    # 构建Markdown内容
    markdown_content = f"""# AI股票分析报告

**生成时间**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}
**分析系统**: LangGraph + DeepSeek + LangChain

---

## 📊 数据概览

"""
    
    # 添加股票数据概览
    if not state["stock_data"].empty:
        stock_data = state["stock_data"]
        markdown_content += f"""
### 筛选股票数量: {len(stock_data)} 只

#### 前5只股票概览:
| 代码 | 名称 | 最新价 | 涨跌幅 | 成交量 | 流通市值 |
|------|------|--------|--------|--------|----------|
"""
        for _, row in stock_data.head(30).iterrows():
            markdown_content += f"| {row.get('代码', 'N/A')} | {row.get('名称', 'N/A')} | {row.get('最新价', 'N/A')} | {row.get('涨跌幅', 'N/A')}% | {row.get('成交量', 'N/A')} | {row.get('流通市值', 'N/A')} |\n"
    
    # 添加新闻分析
    markdown_content += "---\n\n## 📰 新闻分析\n\n"
    if "news_data" in state and "analysis" in state["news_data"]:
        news_analysis = state["news_data"]["analysis"]
        markdown_content += f"""
### AI新闻分析结果:

```
{news_analysis}
```

"""
    else:
        markdown_content += "⚠️ 新闻分析暂无数据\n\n"
    
    # 添加技术分析
    markdown_content += "---\n\n## 🔍 技术分析\n\n"
    if "technical_analysis" in state and "analysis" in state["technical_analysis"]:
        technical_analysis = state["technical_analysis"]["analysis"]
        markdown_content += f"""
### AI技术分析结果:

```
{technical_analysis}
```

"""
    else:
        markdown_content += "⚠️ 技术分析暂无数据\n\n"
    
    # 添加基本面分析
    markdown_content += "---\n\n## 📈 基本面分析\n\n"
    if "fundamental_analysis" in state and "analysis" in state["fundamental_analysis"]:
        fundamental_analysis = state["fundamental_analysis"]["analysis"]
        markdown_content += f"""
### AI基本面分析结果:

```
{fundamental_analysis}
```

"""
    else:
        markdown_content += "⚠️ 基本面分析暂无数据\n\n"
    
    # 添加最终投资建议
    markdown_content += "---\n\n## 💡 最终投资建议\n\n"
    if "final_recommendations" in state and "recommendations" in state["final_recommendations"]:
        recommendations = state["final_recommendations"]["recommendations"]
        markdown_content += f"""
### AI综合投资建议:

```json
{recommendations}
```

"""
    else:
        markdown_content += "⚠️ 投资建议暂无数据\n\n"
    
    # 添加风险提示
    markdown_content += """---

## ⚠️ 风险提示

1. **市场风险**: 股票市场存在波动风险，投资需谨慎
2. **AI分析局限性**: AI分析仅供参考，不构成投资建议
3. **数据时效性**: 分析基于当前数据，市场情况可能发生变化
4. **投资者适当性**: 请根据自身风险承受能力进行投资决策

---

## 📝 免责声明

本报告由AI系统自动生成，仅供参考，不构成任何投资建议。投资者应当根据自己的风险承受能力、投资目标和财务状况做出独立的投资决策。投资有风险，入市需谨慎。

---

*报告生成系统: LangGraph + DeepSeek + LangChain*  
*数据来源: akshare*
"""
    
    # 写入Markdown文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Markdown分析报告已生成: {filepath}")
    return filepath

if __name__ == "__main__":
    print("=== AI股票选择分析系统 ===")
    print("使用 LangGraph + DeepSeek + LangChain 进行智能选股分析")
    
    # 检查API密钥
    if DEEPSEEK_API_KEY == "your_deepseek_api_key":
        print("⚠️  请先设置DeepSeek API密钥")
        print("请在代码中将 DEEPSEEK_API_KEY 替换为您的实际API密钥")
        exit(1)
    
    # 测试DeepSeek连接
    if not test_deepseek_connection():
        print("请检查API密钥和网络连接后重试")
        exit(1)
    
    try:
        # 创建工作流
        app = create_stock_analysis_workflow()
        
        # 初始化状态
        initial_state = {
            "messages": [],
            "stock_data": pd.DataFrame(),
            "news_data": {},
            "technical_analysis": {},
            "fundamental_analysis": {},
            "final_recommendations": {}
        }
        
        # 执行工作流
        print("\n开始执行AI分析工作流...")
        final_state = app.invoke(initial_state)
        
        # 保存结果到Excel
        excel_path = save_analysis_to_excel(final_state)
        
        # 生成Markdown报告
        markdown_path = generate_markdown_report(final_state)
        
        # 显示分析摘要
        print("\n=== 分析完成 ===")
        if final_state["final_recommendations"].get("recommendations"):
            print("最终投资建议：")
            print(final_state["final_recommendations"]["recommendations"])
        
        print(f"\n完整分析报告已保存至:")
        print(f"📊 Excel报告: {excel_path}")
        print(f"📝 Markdown报告: {markdown_path}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查API密钥配置和网络连接")