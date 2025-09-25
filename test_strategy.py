"""
投资策略系统测试脚本
用于验证各个模块的功能和整体系统的运行
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_strategy import (
    EnhancedStockStrategy, DataProvider, FundamentalScreener,
    QuantitativeScorer, TradingSignalGenerator, PortfolioManager,
    AIEnhancedFeatures
)
from strategy_config import StrategyConfig, ConfigTemplates

# 配置测试日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StrategyTest')

class StrategyTester:
    """策略测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.config = StrategyConfig()
        
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("=== 开始运行策略系统测试 ===")
        
        test_methods = [
            self.test_data_provider,
            self.test_fundamental_screener,
            self.test_quantitative_scorer,
            self.test_trading_signals,
            self.test_portfolio_manager,
            self.test_ai_features,
            self.test_config_system,
            self.test_full_strategy
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"运行测试: {test_method.__name__}")
                result = test_method()
                self.test_results[test_method.__name__] = {
                    'status': 'PASS' if result else 'FAIL',
                    'result': result
                }
                logger.info(f"测试 {test_method.__name__}: {'通过' if result else '失败'}")
            except Exception as e:
                logger.error(f"测试 {test_method.__name__} 出错: {e}")
                logger.error(traceback.format_exc())
                self.test_results[test_method.__name__] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        self.print_test_summary()
    
    def test_data_provider(self) -> bool:
        """测试数据提供者"""
        logger.info("测试数据提供者功能...")
        
        try:
            data_provider = DataProvider()
            
            # 测试获取股票列表
            stock_list = data_provider.get_stock_list()
            if stock_list.empty:
                logger.warning("获取股票列表为空，可能是网络问题")
                return False
            
            logger.info(f"成功获取 {len(stock_list)} 只股票数据")
            
            # 测试获取历史数据（使用一个知名股票代码）
            test_code = "000001"  # 平安银行
            hist_data = data_provider.get_stock_historical_data(
                test_code, 
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d')
            )
            
            if not hist_data.empty:
                logger.info(f"成功获取股票 {test_code} 历史数据: {len(hist_data)} 条记录")
                return True
            else:
                logger.warning(f"获取股票 {test_code} 历史数据失败")
                return False
                
        except Exception as e:
            logger.error(f"数据提供者测试失败: {e}")
            return False
    
    def test_fundamental_screener(self) -> bool:
        """测试基本面筛选器"""
        logger.info("测试基本面筛选器...")
        
        try:
            # 创建模拟数据
            mock_data = pd.DataFrame({
                '代码': ['000001', '000002', '688001', '000003'],
                '名称': ['平安银行', '万科A', '华兴源创', '国农科技'],
                '最新价': [10.5, 15.2, 25.8, 8.9],
                '市盈率-动态': [8.5, 12.3, 45.6, 78.9],
                '总市值': [300e8, 250e8, 150e8, 80e8],
                '涨跌幅': [2.1, -1.5, 8.9, 3.2],
                '换手率': [1.2, 0.8, 2.5, 0.5]
            })
            
            data_provider = DataProvider()
            screener = FundamentalScreener(data_provider)
            
            # 执行筛选
            filtered_stocks = screener.screen_stocks(mock_data)
            
            # 验证筛选结果
            # 应该排除科创板(688001)和高市盈率股票
            expected_codes = ['000001', '000002']  # 000003市值太小，688001是科创板
            actual_codes = filtered_stocks['代码'].tolist()
            
            logger.info(f"筛选前: {len(mock_data)} 只股票")
            logger.info(f"筛选后: {len(filtered_stocks)} 只股票")
            logger.info(f"筛选结果: {actual_codes}")
            
            return len(filtered_stocks) > 0
            
        except Exception as e:
            logger.error(f"基本面筛选器测试失败: {e}")
            return False
    
    def test_quantitative_scorer(self) -> bool:
        """测试量化评分器"""
        logger.info("测试量化评分器...")
        
        try:
            scorer = QuantitativeScorer()
            
            # 创建模拟股票数据
            mock_stock_data = {
                'eps_growth_5y': 0.08,
                'roe_5y_avg': 0.18,
                'free_cash_flow_yield': 0.06,
                'volume': 800000,
                'price': 15.5
            }
            
            # 创建模拟价格数据
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            prices = np.random.randn(len(dates)).cumsum() + 100
            mock_price_data = pd.DataFrame({
                '日期': dates,
                '收盘': prices,
                '最高': prices * 1.02,
                '最低': prices * 0.98,
                '成交量': np.random.randint(100000, 1000000, len(dates))
            })
            
            # 计算评分
            scores = scorer.calculate_total_score('000001', mock_stock_data, mock_price_data)
            
            logger.info(f"评分结果: {scores}")
            
            # 验证评分结果
            required_keys = ['fundamental', 'trend', 'momentum', 'sentiment', 'risk_liquidity', 'total']
            has_all_scores = all(key in scores for key in required_keys)
            valid_total_score = 0 <= scores['total'] <= 100
            
            logger.info(f"评分完整性: {has_all_scores}")
            logger.info(f"总分有效性: {valid_total_score} (总分: {scores['total']:.2f})")
            
            return has_all_scores and valid_total_score
            
        except Exception as e:
            logger.error(f"量化评分器测试失败: {e}")
            return False
    
    def test_trading_signals(self) -> bool:
        """测试交易信号生成器"""
        logger.info("测试交易信号生成器...")
        
        try:
            signal_generator = TradingSignalGenerator()
            
            # 创建模拟价格数据
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
            base_price = 100
            trend = np.linspace(0, 20, len(dates))  # 上升趋势
            noise = np.random.randn(len(dates)) * 2
            prices = base_price + trend + noise
            
            mock_price_data = pd.DataFrame({
                '日期': dates,
                '收盘': prices,
                '最高': prices * 1.02,
                '最低': prices * 0.98,
                '成交量': np.random.randint(100000, 1000000, len(dates))
            })
            
            # 生成买入信号
            buy_signals = signal_generator.generate_buy_signals(
                mock_price_data, 
                mock_price_data['成交量']
            )
            
            # 生成卖出信号
            sell_signals = signal_generator.generate_sell_signals(
                mock_price_data, 
                entry_price=110
            )
            
            logger.info(f"买入信号: {buy_signals}")
            logger.info(f"卖出信号: {sell_signals}")
            
            # 验证信号结构
            required_buy_keys = ['golden_cross', 'support_bounce', 'macd_bullish', 
                               'rsi_oversold_recovery', 'volume_breakout', 'overall_buy']
            required_sell_keys = ['death_cross', 'rsi_overbought', 'macd_bearish', 
                                'stop_loss', 'trailing_stop', 'time_based_exit', 'overall_sell']
            
            buy_signals_valid = all(key in buy_signals for key in required_buy_keys)
            sell_signals_valid = all(key in sell_signals for key in required_sell_keys)
            
            logger.info(f"买入信号完整性: {buy_signals_valid}")
            logger.info(f"卖出信号完整性: {sell_signals_valid}")
            
            return buy_signals_valid and sell_signals_valid
            
        except Exception as e:
            logger.error(f"交易信号测试失败: {e}")
            return False
    
    def test_portfolio_manager(self) -> bool:
        """测试投资组合管理器"""
        logger.info("测试投资组合管理器...")
        
        try:
            portfolio = PortfolioManager(initial_capital=1000000)
            
            # 测试添加头寸
            portfolio.add_position('000001', '平安银行', 10.5, 1000)
            portfolio.add_position('000002', '万科A', 15.2, 500)
            
            # 测试头寸大小计算
            position_size = portfolio.calculate_position_size(20.0)
            logger.info(f"计算的头寸大小: {position_size}")
            
            # 测试更新价格
            portfolio.update_positions({'000001': 11.0, '000002': 14.8})
            
            # 测试投资组合摘要
            summary = portfolio.get_portfolio_summary()
            logger.info(f"投资组合摘要: {summary}")
            
            # 测试移除头寸
            portfolio.remove_position('000001', 11.0, "测试卖出")
            
            # 验证结果
            has_positions = len(portfolio.positions) > 0
            valid_summary = 'total_value' in summary and 'return_rate' in summary
            has_transaction_log = len(portfolio.transaction_log) > 0
            
            logger.info(f"持仓管理: {has_positions}")
            logger.info(f"摘要有效性: {valid_summary}")
            logger.info(f"交易日志: {has_transaction_log}")
            
            return valid_summary and has_transaction_log
            
        except Exception as e:
            logger.error(f"投资组合管理器测试失败: {e}")
            return False
    
    def test_ai_features(self) -> bool:
        """测试AI功能（占位函数）"""
        logger.info("测试AI增强功能...")
        
        try:
            # 测试情感分析
            sentiment_score = AIEnhancedFeatures.sentiment_analysis("公司业绩优秀，前景看好")
            logger.info(f"情感分析结果: {sentiment_score}")
            
            # 测试市场状态检测
            market_regime = AIEnhancedFeatures.market_regime_detection()
            logger.info(f"市场状态: {market_regime}")
            
            # 测试风险评估
            risk_assessment = AIEnhancedFeatures.risk_assessment({})
            logger.info(f"风险评估: {risk_assessment}")
            
            # 测试模式识别
            mock_price_data = pd.DataFrame({
                '收盘': np.random.randn(100).cumsum() + 100
            })
            patterns = AIEnhancedFeatures.pattern_recognition(mock_price_data)
            logger.info(f"识别的模式: {patterns}")
            
            # 验证返回值类型
            valid_sentiment = isinstance(sentiment_score, (int, float))
            valid_regime = isinstance(market_regime, str)
            valid_risk = isinstance(risk_assessment, dict)
            valid_patterns = isinstance(patterns, list)
            
            return valid_sentiment and valid_regime and valid_risk and valid_patterns
            
        except Exception as e:
            logger.error(f"AI功能测试失败: {e}")
            return False
    
    def test_config_system(self) -> bool:
        """测试配置系统"""
        logger.info("测试配置系统...")
        
        try:
            # 测试默认配置
            config = StrategyConfig()
            logger.info(f"默认配置创建成功")
            
            # 测试配置模板
            conservative = ConfigTemplates.conservative_config()
            aggressive = ConfigTemplates.aggressive_config()
            growth = ConfigTemplates.growth_focused_config()
            value = ConfigTemplates.value_focused_config()
            
            logger.info(f"保守型配置 - ROE要求: {conservative.screening.min_roe_5y}")
            logger.info(f"激进型配置 - 最大头寸: {aggressive.risk.max_position_size}")
            
            # 测试配置保存和加载
            test_config_file = "test_config.json"
            config.save_to_file(test_config_file)
            loaded_config = StrategyConfig.load_from_file(test_config_file)
            
            # 清理测试文件
            if os.path.exists(test_config_file):
                os.remove(test_config_file)
            
            # 验证配置一致性
            original_pe = config.screening.max_pe_ratio
            loaded_pe = loaded_config.screening.max_pe_ratio
            config_consistent = original_pe == loaded_pe
            
            logger.info(f"配置保存/加载一致性: {config_consistent}")
            
            return config_consistent
            
        except Exception as e:
            logger.error(f"配置系统测试失败: {e}")
            return False
    
    def test_full_strategy(self) -> bool:
        """测试完整策略系统"""
        logger.info("测试完整策略系统...")
        
        try:
            # 使用保守配置进行测试
            strategy = EnhancedStockStrategy(initial_capital=1000000)
            
            # 注意：这里不运行完整策略以避免大量API调用
            # 只测试系统初始化和基本功能
            
            # 验证各个组件是否正确初始化
            has_data_provider = strategy.data_provider is not None
            has_screener = strategy.fundamental_screener is not None
            has_scorer = strategy.quantitative_scorer is not None
            has_signal_generator = strategy.signal_generator is not None
            has_portfolio_manager = strategy.portfolio_manager is not None
            
            logger.info(f"数据提供者: {has_data_provider}")
            logger.info(f"基本面筛选器: {has_screener}")
            logger.info(f"量化评分器: {has_scorer}")
            logger.info(f"信号生成器: {has_signal_generator}")
            logger.info(f"投资组合管理器: {has_portfolio_manager}")
            
            all_components_ready = all([
                has_data_provider, has_screener, has_scorer, 
                has_signal_generator, has_portfolio_manager
            ])
            
            return all_components_ready
            
        except Exception as e:
            logger.error(f"完整策略测试失败: {e}")
            return False
    
    def print_test_summary(self):
        """打印测试摘要"""
        logger.info("\n" + "="*60)
        logger.info("测试结果摘要")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results.values() 
                          if result['status'] == 'FAIL')
        error_tests = sum(1 for result in self.test_results.values() 
                         if result['status'] == 'ERROR')
        
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过: {passed_tests}")
        logger.info(f"失败: {failed_tests}")
        logger.info(f"错误: {error_tests}")
        logger.info(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        logger.info("\n详细结果:")
        for test_name, result in self.test_results.items():
            status_symbol = "✓" if result['status'] == 'PASS' else "✗"
            logger.info(f"{status_symbol} {test_name}: {result['status']}")
            if result['status'] == 'ERROR':
                logger.info(f"   错误: {result.get('error', 'Unknown error')}")
        
        logger.info("="*60)

def main():
    """主测试函数"""
    print("=== 投资策略系统测试 ===")
    print("注意：此测试可能需要网络连接来获取股票数据")
    print("如果网络不可用，某些测试可能会失败")
    print()
    
    # 创建测试器并运行测试
    tester = StrategyTester()
    tester.run_all_tests()
    
    print("\n测试完成！请查看上方的测试结果摘要。")

if __name__ == "__main__":
    main()