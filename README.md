# 增强股票投资策略系统

基于akshare数据的comprehensive股票投资策略实现，集成了现代量化投资理念和风险管理技术。

## 🎯 系统特性

### 核心功能
- **增强基本面筛选** - 更广泛的指标集，包括流动性、现金流和ESG因素
- **自适应加权评分模型** - 动态权重调整和历史回测验证
- **多确认技术框架** - 增强的风险控制和止损机制
- **现代工具整合** - sentiment分析和多样化数据源（占位实现）
- **投资组合级风险管理** - 多样化和行为防护机制

### 技术亮点
- 基于akshare的实时数据获取
- 模块化设计，易于扩展和维护
- 完整的配置系统，支持多种投资风格
- 详细的日志记录和结果保存
- 全面的测试覆盖

## 📁 文件结构

```
btc/
├── new_strategy.py          # 主策略系统实现
├── strategy_config.py       # 配置管理系统
├── test_strategy.py         # 测试脚本
├── README.md               # 使用说明（本文件）
├── filter_stocks.py        # 原有的股票筛选脚本
├── my_stock.py             # 股票报告获取
└── 其他辅助文件...
```

## 🚀 快速开始

### 环境要求

```bash
pip install pandas numpy akshare warnings datetime typing json os dataclasses abc logging concurrent.futures
```

### 基本使用

```python
from new_strategy import EnhancedStockStrategy
from strategy_config import StrategyConfig, ConfigTemplates

# 1. 使用默认配置
strategy = EnhancedStockStrategy(initial_capital=1000000)
results = strategy.run_full_strategy()

# 2. 使用预定义配置模板
conservative_config = ConfigTemplates.conservative_config()
strategy_conservative = EnhancedStockStrategy(initial_capital=1000000)
# 注意：当前版本需要手动应用配置，后续版本将支持配置注入

# 3. 查看结果
if results['status'] == 'success':
    print(f"筛选出 {results['selected_stocks_count']} 只股票")
    print(f"投资组合收益率: {results['portfolio_summary']['return_rate']:.2%}")
```

### 运行测试

```bash
python test_strategy.py
```

## 📊 策略详解

### 第一步：构建观察池 - 增强基本面筛选

筛选标准包括：

#### 盈利能力
- 至少5年连续正EPS，复合年增长率 > 5%
- ROE五年平均 > 15%，标准差 < 5%
- 股息收益率 > 2% 或支付比率 < 60%

#### 财务健康
- 债务权益比 < 0.5
- 流动比率 > 1.5
- 日均成交量 > 50万股

#### 估值合理性
- 市盈率在5-50之间
- 价格/自由现金流 < 20
- 总市值 > 200亿

#### ESG因素
- ESG评分 > 50/100（占位实现）

### 第二步：量化评估 - 自适应加权评分模型

评分维度和权重：

| 维度 | 权重 | 主要指标 |
|------|------|----------|
| 基本面强度 | 35% | EPS增长率、ROE、自由现金流收益率 |
| 长期趋势 | 25% | 200日均线位置、Beta系数 |
| 中期动量 | 20% | 50日均线、ATR波动率 |
| 新闻情感 | 10% | sentiment分析（占位） |
| 风险流动性 | 10% | 成交量、Sharpe比率 |

### 第三步：进出场时机 - 多确认技术框架

#### 买入信号
- 黄金交叉或50日均线支撑反弹
- MACD看涨交叉确认
- RSI从超卖区域回升（40-60）
- 成交量突破（>20日均量150%）
- 至少2个信号确认

#### 卖出信号
- 死亡交叉或跌破200日均线
- RSI超买（>70）或MACD看跌背离
- 止损：10%固定止损
- 跟踪止损：15%动态回撤
- 至少2个信号确认或触发止损

### 第四步：实施和持续改进

#### 资金分配规则
- 单个头寸最多5%
- 单个行业暴露不超过20%
- 单笔交易风险<1%投资组合
- 最多持有20个头寸

#### 风险控制
- 投资组合级止损：总回撤>10%
- 强制24小时冷却期
- 定期重新平衡（月度）

## ⚙️ 配置系统

### 预定义配置模板

```python
from strategy_config import ConfigTemplates

# 保守型配置 - 更严格的筛选和风险控制
conservative = ConfigTemplates.conservative_config()

# 激进型配置 - 放宽条件，更高风险收益
aggressive = ConfigTemplates.aggressive_config()

# 成长型配置 - 重视成长性指标
growth = ConfigTemplates.growth_focused_config()

# 价值型配置 - 重视估值和股息
value = ConfigTemplates.value_focused_config()
```

### 自定义配置

```python
from strategy_config import StrategyConfig

config = StrategyConfig()

# 修改筛选条件
config.screening.min_roe_5y = 0.20  # 提高ROE要求
config.screening.max_pe_ratio = 30   # 降低PE上限

# 调整评分权重
config.scoring.fundamental_weight = 0.40
config.scoring.trend_weight = 0.30

# 修改风险参数
config.risk.max_position_size = 0.03
config.risk.stop_loss_pct = 0.08

# 保存配置
config.save_to_file("my_config.json")
```

## 📈 结果输出

系统会生成以下文件：

- `selected_stocks_YYYYMMDD_HHMMSS.csv` - 筛选出的股票列表
- `trading_signals_YYYYMMDD_HHMMSS.json` - 交易信号详情
- `portfolio_summary_YYYYMMDD_HHMMSS.json` - 投资组合摘要
- `strategy.log` - 详细运行日志

### 示例输出

```
=== 策略执行成功！===
筛选出股票数量: 25
生成交易信号数量: 10
投资组合摘要:
初始资金: ¥1,000,000.00
当前现金: ¥850,000.00
总资产价值: ¥1,050,000.00
总盈亏: ¥50,000.00
收益率: 5.00%
持仓数量: 8

前10只推荐股票:
 1. 贵州茅台(600519) - 评分: 92.5, 价格: ¥1,680.00, 市值: 2,112亿
 2. 宁德时代(300750) - 评分: 89.3, 价格: ¥185.50, 市值: 8,156亿
 ...
```

## 🧪 测试系统

运行完整测试套件：

```bash
python test_strategy.py
```

测试覆盖：
- ✓ 数据提供者功能
- ✓ 基本面筛选器
- ✓ 量化评分系统
- ✓ 交易信号生成
- ✓ 投资组合管理
- ✓ AI功能占位
- ✓ 配置系统
- ✓ 完整策略集成

## 🔧 扩展开发

### 添加新的筛选条件

```python
class CustomScreener(FundamentalScreener):
    def screen_stocks(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        # 调用父类筛选
        filtered = super().screen_stocks(stock_list)
        
        # 添加自定义条件
        custom_filtered = filtered[
            filtered['自定义指标'] > 某个阈值
        ]
        
        return custom_filtered
```

### 集成真实的AI功能

```python
class RealAIFeatures(AIEnhancedFeatures):
    @staticmethod
    def sentiment_analysis(news_text: str) -> float:
        # 集成FinBERT或其他NLP模型
        from transformers import pipeline
        classifier = pipeline("sentiment-analysis", 
                            model="ProsusAI/finbert")
        result = classifier(news_text)
        return result[0]['score']
```

### 添加新的技术指标

```python
class ExtendedTechnicalAnalyzer(TechnicalAnalyzer):
    @staticmethod
    def calculate_custom_indicator(prices: pd.Series) -> pd.Series:
        # 实现自定义技术指标
        return prices.rolling(20).apply(lambda x: custom_calculation(x))
```

## ⚠️ 重要说明

### 免责声明
- 本系统仅供学习和研究使用
- 不构成任何投资建议
- 投资有风险，决策需谨慎
- 历史表现不代表未来收益

### 数据依赖
- 依赖akshare库获取股票数据
- 需要稳定的网络连接
- API调用可能有频率限制
- 建议在非交易时间运行以减少延迟

### 性能考虑
- 完整策略运行可能需要较长时间
- 建议使用缓存机制减少重复API调用
- 可以通过调整筛选条件来控制处理的股票数量

## 🔄 版本历史

### v1.0.0 (当前版本)
- ✅ 完整的策略框架实现
- ✅ 模块化设计和配置系统
- ✅ 基本面筛选和量化评分
- ✅ 技术分析和交易信号
- ✅ 投资组合管理和风险控制
- ✅ AI功能占位实现
- ✅ 全面的测试覆盖

### 计划中的功能
- 🔄 真实AI功能集成
- 🔄 完整的回测引擎
- 🔄 实时监控和预警
- 🔄 Web界面和可视化
- 🔄 更多技术指标和策略

## 📞 支持

如有问题或建议，请：
1. 查看日志文件 `strategy.log`
2. 运行测试脚本检查系统状态
3. 检查网络连接和akshare库版本
4. 参考配置文档调整参数

---

**祝您投资顺利！** 📈