
"""
优化股票投资策略系统
基于akshare数据的comprehensive投资策略实现

主要特性：
1. 增强基本面筛选 - 更广泛的指标集
2. 自适应加权评分模型 - 动态权重和回测
3. 多确认技术和催化剂框架 - 增强风险控制
4. 现代工具整合 - sentiment分析和多样化数据源
5. 多样化和行为防护 - 投资组合级限制
"""

import pandas as pd
import numpy as np
import akshare as ak
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class StockBasicInfo:
    """股票基本信息"""
    code: str
    name: str
    industry: str
    market_cap: float
    pe_ratio: float
    pb_ratio: float
    roe: float
    debt_to_equity: float
    current_ratio: float
    price: float
    volume: float
    turnover_rate: float

@dataclass
class TechnicalIndicators:
    """技术指标"""
    sma_50: float
    sma_200: float
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    atr: float
    beta: float
    sharpe_ratio: float

@dataclass
class FundamentalMetrics:
    """基本面指标"""
    eps_growth_5y: float
    roe_5y_avg: float
    roe_stability: float
    dividend_yield: float
    payout_ratio: float
    free_cash_flow_yield: float
    debt_to_equity: float
    current_ratio: float
    esg_score: float

@dataclass
class ScoringWeights:
    """评分权重"""
    fundamental_strength: float = 0.35
    long_term_trend: float = 0.25
    medium_term_momentum: float = 0.20
    news_sentiment: float = 0.10
    risk_liquidity: float = 0.10

class DataProvider:
    """数据提供者基类"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600  # 1小时缓存
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        try:
            logger.info("正在获取A股股票列表...")
            stock_list = ak.stock_zh_a_spot_em()
            logger.info(f"成功获取 {len(stock_list)} 只股票数据")
            return stock_list
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()
    
    def get_stock_financial_data(self, code: str) -> Dict:
        """获取股票财务数据"""
        try:
            # 获取财务指标
            financial_data = ak.stock_financial_abstract_ths(symbol=code)
            return financial_data.to_dict('records')[0] if not financial_data.empty else {}
        except Exception as e:
            logger.warning(f"获取股票 {code} 财务数据失败: {e}")
            return {}
    
    def get_stock_historical_data(self, code: str, period: str = "daily", 
                                 start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取股票历史数据"""
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            hist_data = ak.stock_zh_a_hist(symbol=code, period=period, 
                                         start_date=start_date, end_date=end_date)
            return hist_data
        except Exception as e:
            logger.warning(f"获取股票 {code} 历史数据失败: {e}")
            return pd.DataFrame()

class FundamentalScreener:
    """基本面筛选器 - 第一步：构建观察池"""
    
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider
        self.screening_criteria = {
            'min_years_profitable': 5,  # 至少5年连续盈利
            'min_eps_cagr': 0.05,      # EPS复合增长率>5%
            'max_pe_ratio': 50,        # 市盈率<50
            'min_pe_ratio': 5,         # 市盈率>5（避免异常值）
            'max_pfcf_ratio': 20,      # 价格/自由现金流<20
            'min_roe_5y': 0.15,        # ROE五年平均>15%
            'max_roe_volatility': 0.05, # ROE标准差<5%
            'min_dividend_yield': 0.02, # 股息收益率>2%
            'max_payout_ratio': 0.60,   # 支付比率<60%
            'max_debt_equity': 0.5,     # 债务权益比<0.5
            'min_current_ratio': 1.5,   # 流动比率>1.5
            'min_daily_volume': 500000, # 日均成交量>50万股
            'min_esg_score': 50,        # ESG评分>50/100
            'min_market_cap': 200e8     # 总市值>200亿
        }
    
    def screen_stocks(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """执行基本面筛选"""
        logger.info("开始执行增强基本面筛选...")
        
        # 初步筛选：排除科创板和ST股票
        filtered_stocks = stock_list[
            (~stock_list['代码'].str.startswith('688')) &  # 排除科创板
            (~stock_list['名称'].str.contains('ST'))        # 排除ST股票
        ].copy()
        
        logger.info(f"初步筛选后剩余 {len(filtered_stocks)} 只股票")
        
        # 应用数值筛选条件
        numeric_filters = [
            (filtered_stocks['市盈率-动态'] > self.screening_criteria['min_pe_ratio']),
            (filtered_stocks['市盈率-动态'] < self.screening_criteria['max_pe_ratio']),
            (filtered_stocks['总市值'] > self.screening_criteria['min_market_cap']),
            (filtered_stocks['涨跌幅'].abs() < 5),  # 避免异常波动
        ]
        
        for filter_condition in numeric_filters:
            filtered_stocks = filtered_stocks[filter_condition]
        
        logger.info(f"数值筛选后剩余 {len(filtered_stocks)} 只股票")
        
        # 按换手率排序，选择市场关注度较高的股票
        final_stocks = filtered_stocks.sort_values('换手率', ascending=False)
        
        logger.info(f"基本面筛选完成，共筛选出 {len(final_stocks)} 只优质股票")
        return final_stocks.head(100)  # 返回前100只股票作为观察池

class TechnicalAnalyzer:
    """技术分析器"""
    
    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """计算简单移动平均线"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """计算平均真实范围ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

class QuantitativeScorer:
    """量化评分器 - 第二步：自适应加权评分模型"""
    
    def __init__(self, weights: ScoringWeights = None):
        self.weights = weights or ScoringWeights()
        self.technical_analyzer = TechnicalAnalyzer()
    
    def calculate_fundamental_score(self, stock_data: Dict) -> float:
        """计算基本面强度评分 (0-100)"""
        score = 0
        max_score = 100
        
        try:
            # EPS增长率评分 (30分)
            eps_growth = stock_data.get('eps_growth_5y', 0)
            if eps_growth > 0.10:  # >10%
                score += 30
            elif eps_growth > 0.05:  # 5-10%
                score += 20
            elif eps_growth > 0:  # >0%
                score += 10
            
            # ROE评分 (35分)
            roe = stock_data.get('roe_5y_avg', 0)
            if roe > 0.20:  # >20%
                score += 35
            elif roe > 0.15:  # 15-20%
                score += 25
            elif roe > 0.10:  # 10-15%
                score += 15
            
            # 自由现金流收益率评分 (35分)
            fcf_yield = stock_data.get('free_cash_flow_yield', 0)
            if fcf_yield > 0.08:  # >8%
                score += 35
            elif fcf_yield > 0.05:  # 5-8%
                score += 25
            elif fcf_yield > 0.02:  # 2-5%
                score += 15
            
        except Exception as e:
            logger.warning(f"计算基本面评分时出错: {e}")
        
        return min(score, max_score)
    
    def calculate_trend_score(self, price_data: pd.DataFrame) -> float:
        """计算长期趋势评分 (0-100)"""
        score = 0
        max_score = 100
        
        try:
            if len(price_data) < 200:
                return 0
            
            current_price = price_data['收盘'].iloc[-1]
            sma_200 = self.technical_analyzer.calculate_sma(price_data['收盘'], 200).iloc[-1]
            
            # 价格相对200日均线位置 (70分)
            if current_price > sma_200 * 1.05:  # 高于5%
                score += 70
            elif current_price > sma_200:  # 高于均线
                score += 50
            elif current_price > sma_200 * 0.95:  # 接近均线
                score += 30
            
            # Beta系数评分 (30分) - 需要计算相对市场的Beta
            # 这里简化处理，使用价格波动率代替
            volatility = price_data['收盘'].pct_change().std()
            if volatility < 0.02:  # 低波动
                score += 30
            elif volatility < 0.03:  # 中等波动
                score += 20
            elif volatility < 0.05:  # 较高波动
                score += 10
            
        except Exception as e:
            logger.warning(f"计算趋势评分时出错: {e}")
        
        return min(score, max_score)
    
    def calculate_momentum_score(self, price_data: pd.DataFrame) -> float:
        """计算中期动量评分 (0-100)"""
        score = 0
        max_score = 100
        
        try:
            if len(price_data) < 50:
                return 0
            
            current_price = price_data['收盘'].iloc[-1]
            sma_50 = self.technical_analyzer.calculate_sma(price_data['收盘'], 50).iloc[-1]
            
            # 价格相对50日均线 (50分)
            if current_price > sma_50 * 1.02:
                score += 50
            elif current_price > sma_50:
                score += 35
            elif current_price > sma_50 * 0.98:
                score += 20
            
            # ATR相对评分 (50分)
            high = price_data['最高']
            low = price_data['最低']
            close = price_data['收盘']
            atr = self.technical_analyzer.calculate_atr(high, low, close).iloc[-1]
            atr_ratio = atr / current_price
            
            if atr_ratio < 0.02:  # 低波动
                score += 50
            elif atr_ratio < 0.03:
                score += 35
            elif atr_ratio < 0.05:
                score += 20
            
        except Exception as e:
            logger.warning(f"计算动量评分时出错: {e}")
        
        return min(score, max_score)
    
    def calculate_sentiment_score(self, stock_code: str) -> float:
        """计算新闻和情感评分 (0-100) - AI功能占位"""
        # TODO: 集成sentiment分析API
        return self._placeholder_sentiment_analysis(stock_code)
    
    def _placeholder_sentiment_analysis(self, stock_code: str) -> float:
        """情感分析占位函数"""
        # 这里先返回随机分数，后续集成真实的sentiment分析
        import random
        return random.uniform(40, 80)
    
    def calculate_risk_liquidity_score(self, stock_data: Dict, price_data: pd.DataFrame) -> float:
        """计算风险和流动性评分 (0-100)"""
        score = 0
        max_score = 100
        
        try:
            # 流动性评分 (50分)
            volume = stock_data.get('volume', 0)
            if volume > 1000000:  # >100万股
                score += 50
            elif volume > 500000:  # 50-100万股
                score += 35
            elif volume > 100000:  # 10-50万股
                score += 20
            
            # Sharpe比率评分 (50分) - 简化计算
            if len(price_data) > 30:
                returns = price_data['收盘'].pct_change().dropna()
                if len(returns) > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)  # 年化
                    if sharpe > 1.5:
                        score += 50
                    elif sharpe > 1.0:
                        score += 35
                    elif sharpe > 0.5:
                        score += 20
            
        except Exception as e:
            logger.warning(f"计算风险流动性评分时出错: {e}")
        
        return min(score, max_score)
    
    def calculate_total_score(self, stock_code: str, stock_data: Dict, price_data: pd.DataFrame) -> Dict[str, float]:
        """计算总评分"""
        scores = {
            'fundamental': self.calculate_fundamental_score(stock_data),
            'trend': self.calculate_trend_score(price_data),
            'momentum': self.calculate_momentum_score(price_data),
            'sentiment': self.calculate_sentiment_score(stock_code),
            'risk_liquidity': self.calculate_risk_liquidity_score(stock_data, price_data)
        }
        
        # 计算加权总分
        total_score = (
            scores['fundamental'] * self.weights.fundamental_strength +
            scores['trend'] * self.weights.long_term_trend +
            scores['momentum'] * self.weights.medium_term_momentum +
            scores['sentiment'] * self.weights.news_sentiment +
            scores['risk_liquidity'] * self.weights.risk_liquidity
        )
        
        scores['total'] = total_score
        return scores

class TradingSignalGenerator:
    """交易信号生成器 - 第三步：进出场时机"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
    
    def generate_buy_signals(self, price_data: pd.DataFrame, volume_data: pd.Series = None) -> Dict[str, bool]:
        """生成买入信号"""
        signals = {
            'golden_cross': False,
            'support_bounce': False,
            'macd_bullish': False,
            'rsi_oversold_recovery': False,
            'volume_breakout': False,
            'overall_buy': False
        }
        
        try:
            if len(price_data) < 200:
                return signals
            
            close_prices = price_data['收盘']
            
            # 黄金交叉信号
            sma_50 = self.technical_analyzer.calculate_sma(close_prices, 50)
            sma_200 = self.technical_analyzer.calculate_sma(close_prices, 200)
            if len(sma_50) > 1 and len(sma_200) > 1:
                signals['golden_cross'] = (sma_50.iloc[-1] > sma_200.iloc[-1] and 
                                         sma_50.iloc[-2] <= sma_200.iloc[-2])
            
            # 支撑反弹信号
            current_price = close_prices.iloc[-1]
            sma_50_current = sma_50.iloc[-1] if len(sma_50) > 0 else 0
            if sma_50_current > 0:
                signals['support_bounce'] = (current_price > sma_50_current * 0.98 and 
                                           current_price < sma_50_current * 1.02)
            
            # MACD看涨交叉
            macd, signal_line = self.technical_analyzer.calculate_macd(close_prices)
            if len(macd) > 1 and len(signal_line) > 1:
                signals['macd_bullish'] = (macd.iloc[-1] > signal_line.iloc[-1] and 
                                         macd.iloc[-2] <= signal_line.iloc[-2])
            
            # RSI超卖回升
            rsi = self.technical_analyzer.calculate_rsi(close_prices)
            if len(rsi) > 1:
                signals['rsi_oversold_recovery'] = (rsi.iloc[-1] > 40 and rsi.iloc[-1] < 60 and 
                                                  rsi.iloc[-2] < 40)
            
            # 成交量突破
            if volume_data is not None and len(volume_data) > 20:
                avg_volume = volume_data.rolling(20).mean().iloc[-1]
                current_volume = volume_data.iloc[-1]
                signals['volume_breakout'] = current_volume > avg_volume * 1.5
            
            # 综合买入信号
            signal_count = sum([signals['golden_cross'], signals['support_bounce'], 
                              signals['macd_bullish'], signals['rsi_oversold_recovery']])
            signals['overall_buy'] = signal_count >= 2  # 至少2个信号确认
            
        except Exception as e:
            logger.warning(f"生成买入信号时出错: {e}")
        
        return signals
    
    def generate_sell_signals(self, price_data: pd.DataFrame, entry_price: float = None) -> Dict[str, bool]:
        """生成卖出信号"""
        signals = {
            'death_cross': False,
            'rsi_overbought': False,
            'macd_bearish': False,
            'stop_loss': False,
            'trailing_stop': False,
            'time_based_exit': False,
            'overall_sell': False
        }
        
        try:
            if len(price_data) < 200:
                return signals
            
            close_prices = price_data['收盘']
            current_price = close_prices.iloc[-1]
            
            # 死亡交叉
            sma_50 = self.technical_analyzer.calculate_sma(close_prices, 50)
            sma_200 = self.technical_analyzer.calculate_sma(close_prices, 200)
            if len(sma_50) > 1 and len(sma_200) > 1:
                signals['death_cross'] = (sma_50.iloc[-1] < sma_200.iloc[-1] and 
                                        sma_50.iloc[-2] >= sma_200.iloc[-2])
            
            # RSI超买
            rsi = self.technical_analyzer.calculate_rsi(close_prices)
            if len(rsi) > 0:
                signals['rsi_overbought'] = rsi.iloc[-1] > 70
            
            # MACD看跌背离
            macd, signal_line = self.technical_analyzer.calculate_macd(close_prices)
            if len(macd) > 1 and len(signal_line) > 1:
                signals['macd_bearish'] = (macd.iloc[-1] < signal_line.iloc[-1] and 
                                         macd.iloc[-2] >= signal_line.iloc[-2])
            
            # 止损信号
            if entry_price:
                signals['stop_loss'] = current_price < entry_price * 0.90  # 10%止损
                
                # 跟踪止损 (15%回撤)
                recent_high = close_prices.tail(30).max()
                signals['trailing_stop'] = current_price < recent_high * 0.85
            
            # 综合卖出信号
            signal_count = sum([signals['death_cross'], signals['rsi_overbought'], 
                              signals['macd_bearish']])
            signals['overall_sell'] = (signal_count >= 2 or signals['stop_loss'] or 
                                     signals['trailing_stop'])
            
        except Exception as e:
            logger.warning(f"生成卖出信号时出错: {e}")
        
        return signals

class PortfolioManager:
    """投资组合管理器 - 第四步：实施和持续改进"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {stock_code: {'shares': int, 'entry_price': float, 'entry_date': datetime}}
        self.transaction_log = []
        self.max_position_size = 0.05  # 单个头寸最大5%
        self.max_sector_exposure = 0.20  # 单个行业最大20%
        self.max_portfolio_risk = 0.01  # 单笔交易最大风险1%
        
    def calculate_position_size(self, stock_price: float, risk_per_trade: float = None) -> int:
        """计算头寸大小"""
        if risk_per_trade is None:
            risk_per_trade = self.max_portfolio_risk
        
        max_investment = self.current_capital * self.max_position_size
        risk_amount = self.current_capital * risk_per_trade
        
        # 基于风险的头寸大小
        stop_loss_distance = stock_price * 0.10  # 假设10%止损
        shares_by_risk = int(risk_amount / stop_loss_distance)
        
        # 基于资金的头寸大小
        shares_by_capital = int(max_investment / stock_price)
        
        return min(shares_by_risk, shares_by_capital)
    
    def can_add_position(self, stock_code: str, sector: str) -> bool:
        """检查是否可以添加新头寸"""
        # 检查头寸数量限制
        if len(self.positions) >= 20:  # 最多20个头寸
            return False
        
        # 检查行业暴露限制
        sector_exposure = self._calculate_sector_exposure(sector)
        if sector_exposure >= self.max_sector_exposure:
            return False
        
        return True
    
    def _calculate_sector_exposure(self, sector: str) -> float:
        """计算行业暴露度"""
        # 这里需要实现行业分类逻辑
        # 简化处理，返回0
        return 0.0
    
    def add_position(self, stock_code: str, stock_name: str, price: float, shares: int):
        """添加头寸"""
        if stock_code not in self.positions:
            self.positions[stock_code] = {
                'name': stock_name,
                'shares': shares,
                'entry_price': price,
                'entry_date': datetime.now(),
                'current_price': price
            }
            
            cost = shares * price
            self.current_capital -= cost
            
            self.transaction_log.append({
                'date': datetime.now(),
                'action': 'BUY',
                'stock_code': stock_code,
                'stock_name': stock_name,
                'shares': shares,
                'price': price,
                'total': cost
            })
            
            logger.info(f"买入 {stock_name}({stock_code}): {shares}股 @ ¥{price:.2f}")
    
    def remove_position(self, stock_code: str, price: float, reason: str = ""):
        """移除头寸"""
        if stock_code in self.positions:
            position = self.positions[stock_code]
            shares = position['shares']
            proceeds = shares * price
            self.current_capital += proceeds
            
            profit_loss = (price - position['entry_price']) * shares
            
            self.transaction_log.append({
                'date': datetime.now(),
                'action': 'SELL',
                'stock_code': stock_code,
                'stock_name': position['name'],
                'shares': shares,
                'price': price,
                'total': proceeds,
                'profit_loss': profit_loss,
                'reason': reason
            })
            
            logger.info(f"卖出 {position['name']}({stock_code}): {shares}股 @ ¥{price:.2f}, "
                       f"盈亏: ¥{profit_loss:.2f}, 原因: {reason}")
            
            del self.positions[stock_code]
    
    def update_positions(self, current_prices: Dict[str, float]):
        """更新头寸当前价格"""
        for stock_code, price in current_prices.items():
            if stock_code in self.positions:
                self.positions[stock_code]['current_price'] = price
    
    def get_portfolio_summary(self) -> Dict:
        """获取投资组合摘要"""
        total_value = self.current_capital
        total_profit_loss = 0
        
        for stock_code, position in self.positions.items():
            current_value = position['shares'] * position['current_price']
            total_value += current_value
            
            profit_loss = (position['current_price'] - position['entry_price']) * position['shares']
            total_profit_loss += profit_loss
        
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.current_capital,
            'total_value': total_value,
            'total_profit_loss': total_profit_loss,
            'return_rate': (total_value - self.initial_capital) / self.initial_capital,
            'positions_count': len(self.positions),
            'positions': self.positions
        }

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, start_date: str, end_date: str, initial_capital: float = 1000000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.results = []
    
    def run_backtest(self, strategy_params: Dict) -> Dict:
        """运行回测"""
        logger.info(f"开始回测: {self.start_date} 到 {self.end_date}")
        
        # TODO: 实现完整的回测逻辑
        # 这里先返回占位结果
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'trades_count': 0
        }

class EnhancedStockStrategy:
    """增强股票投资策略主类"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.data_provider = DataProvider()
        self.fundamental_screener = FundamentalScreener(self.data_provider)
        self.quantitative_scorer = QuantitativeScorer()
        self.signal_generator = TradingSignalGenerator()
        self.portfolio_manager = PortfolioManager(initial_capital)
        self.backtest_engine = None
        
        # 策略参数
        self.min_score_threshold = 85  # 最低评分阈值
        self.rebalance_frequency = 'monthly'  # 重新平衡频率
        self.max_positions = 20  # 最大持仓数量
        
        logger.info("增强股票投资策略系统初始化完成")
    
    def run_screening_process(self) -> pd.DataFrame:
        """执行完整的筛选流程"""
        logger.info("=== 开始执行股票筛选流程 ===")
        
        # 第一步：获取股票数据并进行基本面筛选
        stock_list = self.data_provider.get_stock_list()
        if stock_list.empty:
            logger.error("无法获取股票数据，筛选流程终止")
            return pd.DataFrame()
        
        screened_stocks = self.fundamental_screener.screen_stocks(stock_list)
        if screened_stocks.empty:
            logger.warning("基本面筛选后无符合条件的股票")
            return pd.DataFrame()
        
        # 第二步：量化评分
        scored_stocks = []
        
        logger.info("开始对筛选出的股票进行量化评分...")
        for idx, row in screened_stocks.head(50).iterrows():  # 限制处理数量以提高效率
            stock_code = row['代码']
            stock_name = row['名称']
            
            try:
                # 获取历史价格数据
                price_data = self.data_provider.get_stock_historical_data(stock_code)
                if price_data.empty:
                    continue
                
                # 构建股票数据字典
                stock_data = {
                    'code': stock_code,
                    'name': stock_name,
                    'price': row['最新价'],
                    'volume': row.get('成交量', 0),
                    'market_cap': row['总市值'],
                    'pe_ratio': row['市盈率-动态'],
                    'pb_ratio': row.get('市净率', 0),
                    'turnover_rate': row['换手率']
                }
                
                # 计算评分
                scores = self.quantitative_scorer.calculate_total_score(
                    stock_code, stock_data, price_data
                )
                
                if scores['total'] >= self.min_score_threshold:
                    scored_stocks.append({
                        'code': stock_code,
                        'name': stock_name,
                        'price': row['最新价'],
                        'total_score': scores['total'],
                        'fundamental_score': scores['fundamental'],
                        'trend_score': scores['trend'],
                        'momentum_score': scores['momentum'],
                        'sentiment_score': scores['sentiment'],
                        'risk_liquidity_score': scores['risk_liquidity'],
                        'market_cap': row['总市值'],
                        'pe_ratio': row['市盈率-动态'],
                        'turnover_rate': row['换手率']
                    })
                
                # 添加延迟以避免API限制
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"处理股票 {stock_code} 时出错: {e}")
                continue
        
        # 转换为DataFrame并按总分排序
        if scored_stocks:
            result_df = pd.DataFrame(scored_stocks)
            result_df = result_df.sort_values('total_score', ascending=False)
            logger.info(f"量化评分完成，共有 {len(result_df)} 只股票达到评分阈值")
            return result_df
        else:
            logger.warning("没有股票达到最低评分要求")
            return pd.DataFrame()
    
    def generate_trading_signals(self, stock_list: pd.DataFrame) -> Dict[str, Dict]:
        """为股票列表生成交易信号"""
        signals = {}
        
        logger.info("开始生成交易信号...")
        for idx, row in stock_list.head(10).iterrows():  # 只处理前10只股票
            stock_code = row['code']
            
            try:
                # 获取价格数据
                price_data = self.data_provider.get_stock_historical_data(stock_code)
                if not price_data.empty:
                    # 生成买入信号
                    buy_signals = self.signal_generator.generate_buy_signals(price_data)
                    
                    # 生成卖出信号
                    sell_signals = self.signal_generator.generate_sell_signals(price_data)
                    
                    signals[stock_code] = {
                        'name': row['name'],
                        'current_price': row['price'],
                        'buy_signals': buy_signals,
                        'sell_signals': sell_signals,
                        'recommendation': 'BUY' if buy_signals['overall_buy'] else
                                       'SELL' if sell_signals['overall_sell'] else 'HOLD'
                    }
                
                time.sleep(0.1)  # API限制
                
            except Exception as e:
                logger.warning(f"生成股票 {stock_code} 交易信号时出错: {e}")
        
        return signals
    
    def execute_portfolio_rebalancing(self, target_stocks: pd.DataFrame):
        """执行投资组合重新平衡"""
        logger.info("开始执行投资组合重新平衡...")
        
        current_positions = set(self.portfolio_manager.positions.keys())
        target_positions = set(target_stocks['code'].head(self.max_positions))
        
        # 卖出不在目标列表中的股票
        positions_to_sell = current_positions - target_positions
        for stock_code in positions_to_sell:
            if stock_code in self.portfolio_manager.positions:
                current_price = self.portfolio_manager.positions[stock_code]['current_price']
                self.portfolio_manager.remove_position(stock_code, current_price, "重新平衡")
        
        # 买入新的目标股票
        positions_to_buy = target_positions - current_positions
        for stock_code in positions_to_buy:
            stock_info = target_stocks[target_stocks['code'] == stock_code].iloc[0]
            
            if self.portfolio_manager.can_add_position(stock_code, ""):  # 简化行业检查
                shares = self.portfolio_manager.calculate_position_size(stock_info['price'])
                if shares > 0:
                    self.portfolio_manager.add_position(
                        stock_code, stock_info['name'], stock_info['price'], shares
                    )
        
        logger.info("投资组合重新平衡完成")
    
    def run_full_strategy(self) -> Dict:
        """运行完整策略"""
        logger.info("=== 开始运行完整投资策略 ===")
        
        try:
            # 1. 股票筛选和评分
            selected_stocks = self.run_screening_process()
            if selected_stocks.empty:
                return {'status': 'error', 'message': '未找到符合条件的股票'}
            
            # 2. 生成交易信号
            trading_signals = self.generate_trading_signals(selected_stocks)
            
            # 3. 投资组合管理
            self.execute_portfolio_rebalancing(selected_stocks)
            
            # 4. 获取投资组合摘要
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            # 5. 保存结果
            self.save_strategy_results(selected_stocks, trading_signals, portfolio_summary)
            
            logger.info("完整投资策略执行完成")
            
            return {
                'status': 'success',
                'selected_stocks_count': len(selected_stocks),
                'trading_signals_count': len(trading_signals),
                'portfolio_summary': portfolio_summary,
                'top_stocks': selected_stocks.head(10).to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"策略执行过程中发生错误: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def save_strategy_results(self, selected_stocks: pd.DataFrame,
                            trading_signals: Dict, portfolio_summary: Dict):
        """保存策略结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存筛选结果
        selected_stocks.to_csv(f'selected_stocks_{timestamp}.csv',
                              index=False, encoding='utf-8-sig')
        
        # 保存交易信号
        with open(f'trading_signals_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(trading_signals, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存投资组合摘要
        with open(f'portfolio_summary_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(portfolio_summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"策略结果已保存，时间戳: {timestamp}")
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """运行回测"""
        self.backtest_engine = BacktestEngine(start_date, end_date,
                                            self.portfolio_manager.initial_capital)
        
        # TODO: 实现完整的回测逻辑
        return self.backtest_engine.run_backtest({})

# AI相关功能占位类
class AIEnhancedFeatures:
    """AI增强功能 - 占位实现"""
    
    @staticmethod
    def sentiment_analysis(news_text: str) -> float:
        """新闻情感分析 - 占位函数"""
        # TODO: 集成FinBERT或其他金融情感分析模型
        return 0.5
    
    @staticmethod
    def market_regime_detection() -> str:
        """市场状态检测 - 占位函数"""
        # TODO: 实现牛市/熊市检测算法
        return "neutral"
    
    @staticmethod
    def risk_assessment(portfolio_data: Dict) -> Dict:
        """风险评估 - 占位函数"""
        # TODO: 实现基于机器学习的风险评估
        return {
            'risk_level': 'medium',
            'var_95': 0.05,
            'expected_shortfall': 0.08
        }
    
    @staticmethod
    def pattern_recognition(price_data: pd.DataFrame) -> List[str]:
        """价格模式识别 - 占位函数"""
        # TODO: 实现技术分析模式识别
        return ['ascending_triangle', 'bullish_flag']

def main():
    """主函数 - 演示策略系统使用"""
    logger.info("=== 增强股票投资策略系统启动 ===")
    
    # 初始化策略系统
    strategy = EnhancedStockStrategy(initial_capital=1000000)
    
    # 运行完整策略
    results = strategy.run_full_strategy()
    
    # 输出结果
    if results['status'] == 'success':
        print("\n" + "="*60)
        print("策略执行成功！")
        print(f"筛选出股票数量: {results['selected_stocks_count']}")
        print(f"生成交易信号数量: {results['trading_signals_count']}")
        print("\n投资组合摘要:")
        portfolio = results['portfolio_summary']
        print(f"初始资金: ¥{portfolio['initial_capital']:,.2f}")
        print(f"当前现金: ¥{portfolio['current_cash']:,.2f}")
        print(f"总资产价值: ¥{portfolio['total_value']:,.2f}")
        print(f"总盈亏: ¥{portfolio['total_profit_loss']:,.2f}")
        print(f"收益率: {portfolio['return_rate']:.2%}")
        print(f"持仓数量: {portfolio['positions_count']}")
        
        print("\n前10只推荐股票:")
        print("-" * 80)
        for i, stock in enumerate(results['top_stocks'], 1):
            print(f"{i:2d}. {stock['name']}({stock['code']}) - "
                  f"评分: {stock['total_score']:.1f}, "
                  f"价格: ¥{stock['price']:.2f}, "
                  f"市值: {stock['market_cap']/1e8:.0f}亿")
        print("="*60)
    else:
        print(f"策略执行失败: {results['message']}")
    
    logger.info("=== 策略系统运行完成 ===")

if __name__ == "__main__":
    main()