"""
投资策略配置文件
包含所有可调整的策略参数和设置
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ScreeningConfig:
    """基本面筛选配置"""
    # 盈利能力要求
    min_years_profitable: int = 5          # 最少连续盈利年数
    min_eps_cagr: float = 0.05            # EPS最小复合增长率
    min_roe_5y: float = 0.15              # ROE五年平均最小值
    max_roe_volatility: float = 0.05       # ROE波动性上限
    
    # 估值要求
    min_pe_ratio: float = 5               # 市盈率下限
    max_pe_ratio: float = 50              # 市盈率上限
    max_pfcf_ratio: float = 20            # 价格/自由现金流上限
    
    # 股东回报要求
    min_dividend_yield: float = 0.02      # 最小股息收益率
    max_payout_ratio: float = 0.60        # 最大支付比率
    
    # 财务健康要求
    max_debt_equity: float = 0.5          # 最大债务权益比
    min_current_ratio: float = 1.5        # 最小流动比率
    
    # 流动性要求
    min_daily_volume: int = 500000        # 最小日均成交量
    min_market_cap: float = 200e8         # 最小市值（200亿）
    
    # ESG要求
    min_esg_score: float = 50             # 最小ESG评分
    
    # 其他筛选条件
    exclude_st_stocks: bool = True        # 排除ST股票
    exclude_kcb_stocks: bool = True       # 排除科创板
    max_daily_change: float = 0.05        # 最大日涨跌幅

@dataclass
class ScoringConfig:
    """评分系统配置"""
    # 权重配置
    fundamental_weight: float = 0.35      # 基本面权重
    trend_weight: float = 0.25           # 长期趋势权重
    momentum_weight: float = 0.20        # 中期动量权重
    sentiment_weight: float = 0.10       # 情感分析权重
    risk_liquidity_weight: float = 0.10  # 风险流动性权重
    
    # 评分阈值
    min_total_score: float = 85          # 最低总评分
    
    # 基本面评分参数
    excellent_eps_growth: float = 0.10   # 优秀EPS增长率
    good_eps_growth: float = 0.05        # 良好EPS增长率
    excellent_roe: float = 0.20          # 优秀ROE
    good_roe: float = 0.15               # 良好ROE
    excellent_fcf_yield: float = 0.08    # 优秀自由现金流收益率
    good_fcf_yield: float = 0.05         # 良好自由现金流收益率
    
    # 技术分析参数
    sma_trend_threshold: float = 0.05    # 趋势判断阈值
    low_volatility_threshold: float = 0.02  # 低波动阈值
    medium_volatility_threshold: float = 0.03  # 中等波动阈值
    high_volatility_threshold: float = 0.05   # 高波动阈值

@dataclass
class TradingConfig:
    """交易信号配置"""
    # 技术指标参数
    sma_short_period: int = 50           # 短期均线周期
    sma_long_period: int = 200           # 长期均线周期
    rsi_period: int = 14                 # RSI周期
    rsi_oversold: float = 30             # RSI超卖线
    rsi_overbought: float = 70           # RSI超买线
    rsi_recovery_min: float = 40         # RSI回升最小值
    rsi_recovery_max: float = 60         # RSI回升最大值
    
    # MACD参数
    macd_fast: int = 12                  # MACD快线
    macd_slow: int = 26                  # MACD慢线
    macd_signal: int = 9                 # MACD信号线
    
    # 布林带参数
    bollinger_period: int = 20           # 布林带周期
    bollinger_std: float = 2.0           # 布林带标准差倍数
    
    # ATR参数
    atr_period: int = 14                 # ATR周期
    
    # 成交量参数
    volume_breakout_multiplier: float = 1.5  # 成交量突破倍数
    volume_average_period: int = 20      # 成交量平均周期
    
    # 信号确认要求
    min_buy_signals: int = 2             # 最少买入信号数量
    min_sell_signals: int = 2            # 最少卖出信号数量

@dataclass
class RiskConfig:
    """风险管理配置"""
    # 头寸管理
    max_position_size: float = 0.05      # 单个头寸最大占比
    max_sector_exposure: float = 0.20    # 单个行业最大暴露
    max_portfolio_risk: float = 0.01     # 单笔交易最大风险
    max_positions: int = 20              # 最大持仓数量
    
    # 止损设置
    stop_loss_pct: float = 0.10          # 止损百分比
    trailing_stop_pct: float = 0.15      # 跟踪止损百分比
    trailing_stop_period: int = 30       # 跟踪止损观察期
    
    # 投资组合风险
    max_portfolio_drawdown: float = 0.10 # 最大组合回撤
    
    # 行为控制
    cooling_period_days: int = 1         # 交易冷却期（天）
    max_trades_per_day: int = 5          # 每日最大交易次数
    
    # 资金管理
    cash_reserve_ratio: float = 0.05     # 现金储备比例
    rebalance_threshold: float = 0.05    # 重新平衡阈值

@dataclass
class BacktestConfig:
    """回测配置"""
    # 回测期间
    default_start_date: str = "20200101"  # 默认开始日期
    default_end_date: str = "20231231"    # 默认结束日期
    
    # 交易成本
    commission_rate: float = 0.0003      # 佣金费率
    stamp_tax_rate: float = 0.001        # 印花税率（卖出）
    slippage_rate: float = 0.001         # 滑点率
    
    # 基准设置
    benchmark_symbol: str = "000300"     # 基准指数（沪深300）
    
    # 回测频率
    rebalance_frequency: str = "monthly" # 重新平衡频率
    
    # 性能指标
    risk_free_rate: float = 0.03         # 无风险利率

@dataclass
class DataConfig:
    """数据配置"""
    # 缓存设置
    cache_timeout: int = 3600            # 缓存超时时间（秒）
    enable_cache: bool = True            # 是否启用缓存
    
    # API限制
    api_delay: float = 0.1               # API调用延迟（秒）
    max_retries: int = 3                 # 最大重试次数
    
    # 数据质量
    min_trading_days: int = 250          # 最少交易日数据
    max_missing_ratio: float = 0.05      # 最大缺失数据比例
    
    # 文件路径
    log_file: str = "strategy.log"       # 日志文件路径
    data_dir: str = "data"               # 数据目录
    results_dir: str = "results"         # 结果目录

class StrategyConfig:
    """策略总配置类"""
    
    def __init__(self):
        self.screening = ScreeningConfig()
        self.scoring = ScoringConfig()
        self.trading = TradingConfig()
        self.risk = RiskConfig()
        self.backtest = BacktestConfig()
        self.data = DataConfig()
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'screening': self.screening.__dict__,
            'scoring': self.scoring.__dict__,
            'trading': self.trading.__dict__,
            'risk': self.risk.__dict__,
            'backtest': self.backtest.__dict__,
            'data': self.data.__dict__
        }
    
    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """从文件加载配置"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # 更新配置
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        return config

# 预定义配置模板
class ConfigTemplates:
    """配置模板"""
    
    @staticmethod
    def conservative_config() -> StrategyConfig:
        """保守型配置"""
        config = StrategyConfig()
        
        # 更严格的筛选条件
        config.screening.min_roe_5y = 0.18
        config.screening.max_pe_ratio = 30
        config.screening.min_dividend_yield = 0.03
        config.screening.max_debt_equity = 0.3
        
        # 更高的评分要求
        config.scoring.min_total_score = 90
        
        # 更严格的风险控制
        config.risk.max_position_size = 0.03
        config.risk.stop_loss_pct = 0.08
        config.risk.max_positions = 15
        
        return config
    
    @staticmethod
    def aggressive_config() -> StrategyConfig:
        """激进型配置"""
        config = StrategyConfig()
        
        # 放宽筛选条件
        config.screening.min_roe_5y = 0.12
        config.screening.max_pe_ratio = 80
        config.screening.min_dividend_yield = 0.01
        config.screening.max_debt_equity = 0.8
        
        # 降低评分要求
        config.scoring.min_total_score = 75
        
        # 更激进的风险设置
        config.risk.max_position_size = 0.08
        config.risk.stop_loss_pct = 0.15
        config.risk.max_positions = 25
        
        return config
    
    @staticmethod
    def growth_focused_config() -> StrategyConfig:
        """成长型配置"""
        config = StrategyConfig()
        
        # 重视成长性
        config.screening.min_eps_cagr = 0.08
        config.screening.max_pe_ratio = 60
        config.screening.min_dividend_yield = 0.01  # 降低股息要求
        
        # 调整评分权重
        config.scoring.fundamental_weight = 0.40
        config.scoring.momentum_weight = 0.25
        config.scoring.trend_weight = 0.20
        
        return config
    
    @staticmethod
    def value_focused_config() -> StrategyConfig:
        """价值型配置"""
        config = StrategyConfig()
        
        # 重视估值
        config.screening.max_pe_ratio = 25
        config.screening.min_dividend_yield = 0.04
        config.screening.max_debt_equity = 0.4
        
        # 调整评分权重
        config.scoring.fundamental_weight = 0.45
        config.scoring.trend_weight = 0.30
        config.scoring.momentum_weight = 0.15
        
        return config

# 默认配置实例
DEFAULT_CONFIG = StrategyConfig()

if __name__ == "__main__":
    # 演示配置使用
    print("=== 策略配置演示 ===")
    
    # 创建默认配置
    config = StrategyConfig()
    print("默认配置创建完成")
    
    # 保存配置
    config.save_to_file("default_config.json")
    print("配置已保存到 default_config.json")
    
    # 创建不同类型的配置
    conservative = ConfigTemplates.conservative_config()
    aggressive = ConfigTemplates.aggressive_config()
    growth = ConfigTemplates.growth_focused_config()
    value = ConfigTemplates.value_focused_config()
    
    print(f"保守型配置 - 最小ROE: {conservative.screening.min_roe_5y}")
    print(f"激进型配置 - 最大头寸: {aggressive.risk.max_position_size}")
    print(f"成长型配置 - 基本面权重: {growth.scoring.fundamental_weight}")
    print(f"价值型配置 - 最大市盈率: {value.screening.max_pe_ratio}")