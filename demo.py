"""
增强股票投资策略系统演示脚本
展示系统的主要功能和使用方法
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from new_strategy import EnhancedStockStrategy
from strategy_config import StrategyConfig, ConfigTemplates

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StrategyDemo')

def demo_config_system():
    """演示配置系统功能"""
    print("\n" + "="*60)
    print("📋 配置系统演示")
    print("="*60)
    
    # 1. 展示默认配置
    default_config = StrategyConfig()
    print(f"默认配置 - 最小ROE要求: {default_config.screening.min_roe_5y:.1%}")
    print(f"默认配置 - 基本面权重: {default_config.scoring.fundamental_weight:.1%}")
    print(f"默认配置 - 最大头寸大小: {default_config.risk.max_position_size:.1%}")
    
    # 2. 展示不同配置模板
    print("\n📊 配置模板对比:")
    configs = {
        "保守型": ConfigTemplates.conservative_config(),
        "激进型": ConfigTemplates.aggressive_config(),
        "成长型": ConfigTemplates.growth_focused_config(),
        "价值型": ConfigTemplates.value_focused_config()
    }
    
    print(f"{'配置类型':<8} {'ROE要求':<8} {'最大PE':<8} {'头寸大小':<10} {'止损比例':<8}")
    print("-" * 50)
    for name, config in configs.items():
        print(f"{name:<8} {config.screening.min_roe_5y:<8.1%} "
              f"{config.screening.max_pe_ratio:<8.0f} "
              f"{config.risk.max_position_size:<10.1%} "
              f"{config.risk.stop_loss_pct:<8.1%}")
    
    # 3. 演示配置保存和加载
    print("\n💾 配置保存/加载演示:")
    test_config = ConfigTemplates.conservative_config()
    test_config.save_to_file("demo_config.json")
    print("✓ 配置已保存到 demo_config.json")
    
    loaded_config = StrategyConfig.load_from_file("demo_config.json")
    print("✓ 配置已从文件加载")
    print(f"验证: 加载的ROE要求 = {loaded_config.screening.min_roe_5y:.1%}")
    
    # 清理演示文件
    if os.path.exists("demo_config.json"):
        os.remove("demo_config.json")
        print("✓ 演示文件已清理")

def demo_mock_screening():
    """演示筛选功能（使用模拟数据）"""
    print("\n" + "="*60)
    print("🔍 股票筛选演示（模拟数据）")
    print("="*60)
    
    # 创建模拟股票数据
    mock_stocks = pd.DataFrame({
        '代码': ['000001', '000002', '000858', '600519', '300750', '688001', '000003'],
        '名称': ['平安银行', '万科A', '五粮液', '贵州茅台', '宁德时代', '华兴源创', '国农科技'],
        '最新价': [10.5, 15.2, 180.5, 1680.0, 185.5, 25.8, 8.9],
        '市盈率-动态': [8.5, 12.3, 25.6, 35.2, 28.9, 45.6, 78.9],
        '总市值': [300e8, 250e8, 1500e8, 2100e8, 8000e8, 150e8, 80e8],
        '涨跌幅': [2.1, -1.5, 1.2, 0.8, -2.1, 8.9, 3.2],
        '换手率': [1.2, 0.8, 0.6, 0.3, 1.8, 2.5, 0.5],
        '市净率': [0.8, 1.2, 3.5, 12.8, 5.2, 4.1, 2.1]
    })
    
    print("📊 原始股票池:")
    print(mock_stocks[['代码', '名称', '最新价', '市盈率-动态', '总市值']].to_string(index=False))
    
    # 应用筛选条件
    print("\n🎯 应用筛选条件:")
    print("- 排除科创板股票（688开头）")
    print("- 市盈率在5-50之间")
    print("- 总市值>200亿")
    print("- 日涨跌幅<5%")
    
    # 执行筛选
    filtered = mock_stocks[
        (~mock_stocks['代码'].str.startswith('688')) &  # 排除科创板
        (mock_stocks['市盈率-动态'] > 5) &
        (mock_stocks['市盈率-动态'] < 50) &
        (mock_stocks['总市值'] > 200e8) &
        (mock_stocks['涨跌幅'].abs() < 5)
    ]
    
    print(f"\n✅ 筛选结果: {len(filtered)}/{len(mock_stocks)} 只股票通过筛选")
    if not filtered.empty:
        print("\n通过筛选的股票:")
        display_cols = ['代码', '名称', '最新价', '市盈率-动态', '总市值']
        print(filtered[display_cols].to_string(index=False))
    
    return filtered

def demo_scoring_system():
    """演示评分系统（使用模拟数据）"""
    print("\n" + "="*60)
    print("📊 量化评分系统演示")
    print("="*60)
    
    from new_strategy import QuantitativeScorer
    
    scorer = QuantitativeScorer()
    
    # 模拟不同类型的股票数据
    stock_examples = [
        {
            'name': '优质成长股',
            'data': {
                'eps_growth_5y': 0.12,  # 12% EPS增长
                'roe_5y_avg': 0.22,     # 22% ROE
                'free_cash_flow_yield': 0.08,  # 8% FCF收益率
                'volume': 1200000,      # 120万股成交量
                'price': 50.0
            }
        },
        {
            'name': '稳健价值股',
            'data': {
                'eps_growth_5y': 0.06,  # 6% EPS增长
                'roe_5y_avg': 0.18,     # 18% ROE
                'free_cash_flow_yield': 0.06,  # 6% FCF收益率
                'volume': 800000,       # 80万股成交量
                'price': 25.0
            }
        },
        {
            'name': '风险较高股',
            'data': {
                'eps_growth_5y': 0.03,  # 3% EPS增长
                'roe_5y_avg': 0.12,     # 12% ROE
                'free_cash_flow_yield': 0.03,  # 3% FCF收益率
                'volume': 300000,       # 30万股成交量
                'price': 15.0
            }
        }
    ]
    
    # 创建模拟价格数据
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    print("📈 评分结果对比:")
    print(f"{'股票类型':<12} {'基本面':<8} {'趋势':<8} {'动量':<8} {'情感':<8} {'风险':<8} {'总分':<8}")
    print("-" * 70)
    
    for example in stock_examples:
        # 为每个例子创建不同的价格走势
        if '成长' in example['name']:
            # 成长股：上升趋势
            trend = np.linspace(0, 20, len(dates))
            noise = np.random.randn(len(dates)) * 1
        elif '价值' in example['name']:
            # 价值股：稳定走势
            trend = np.linspace(0, 5, len(dates))
            noise = np.random.randn(len(dates)) * 0.5
        else:
            # 风险股：波动走势
            trend = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 5
            noise = np.random.randn(len(dates)) * 2
        
        prices = example['data']['price'] + trend + noise
        mock_price_data = pd.DataFrame({
            '日期': dates,
            '收盘': prices,
            '最高': prices * 1.02,
            '最低': prices * 0.98,
            '成交量': np.random.randint(100000, 1000000, len(dates))
        })
        
        # 计算评分
        scores = scorer.calculate_total_score('demo', example['data'], mock_price_data)
        
        print(f"{example['name']:<12} {scores['fundamental']:<8.1f} "
              f"{scores['trend']:<8.1f} {scores['momentum']:<8.1f} "
              f"{scores['sentiment']:<8.1f} {scores['risk_liquidity']:<8.1f} "
              f"{scores['total']:<8.1f}")
    
    print(f"\n💡 评分说明:")
    print(f"- 基本面评分基于EPS增长、ROE和现金流")
    print(f"- 趋势评分基于长期价格走势和波动性")
    print(f"- 动量评分基于中期价格动量")
    print(f"- 情感评分基于新闻分析（当前为随机值）")
    print(f"- 风险评分基于流动性和风险指标")
    print(f"- 总分≥85分的股票进入最终候选池")

def demo_trading_signals():
    """演示交易信号生成"""
    print("\n" + "="*60)
    print("📡 交易信号生成演示")
    print("="*60)
    
    from new_strategy import TradingSignalGenerator
    
    signal_generator = TradingSignalGenerator()
    
    # 创建不同市场情况的模拟数据
    scenarios = [
        {
            'name': '上升趋势市场',
            'trend': 'bullish',
            'base_price': 100,
            'trend_strength': 0.5
        },
        {
            'name': '下降趋势市场',
            'trend': 'bearish',
            'base_price': 100,
            'trend_strength': -0.3
        },
        {
            'name': '震荡市场',
            'trend': 'sideways',
            'base_price': 100,
            'trend_strength': 0.0
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}:")
        
        # 生成价格数据
        days = 250
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        if scenario['trend'] == 'bullish':
            trend = np.linspace(0, 30, days)
            volatility = 1.5
        elif scenario['trend'] == 'bearish':
            trend = np.linspace(0, -20, days)
            volatility = 2.0
        else:  # sideways
            trend = np.sin(np.linspace(0, 4*np.pi, days)) * 5
            volatility = 1.0
        
        noise = np.random.randn(days) * volatility
        prices = scenario['base_price'] + trend + noise
        
        mock_price_data = pd.DataFrame({
            '日期': dates,
            '收盘': prices,
            '最高': prices * 1.02,
            '最低': prices * 0.98,
            '成交量': np.random.randint(500000, 1500000, days)
        })
        
        # 生成交易信号
        buy_signals = signal_generator.generate_buy_signals(
            mock_price_data, mock_price_data['成交量']
        )
        sell_signals = signal_generator.generate_sell_signals(
            mock_price_data, entry_price=scenario['base_price']
        )
        
        # 显示信号结果
        print(f"   买入信号:")
        for signal, active in buy_signals.items():
            if signal != 'overall_buy':
                status = "✓" if active else "✗"
                print(f"     {status} {signal}")
        
        overall_buy = "🟢 建议买入" if buy_signals['overall_buy'] else "🔴 不建议买入"
        print(f"   综合建议: {overall_buy}")
        
        print(f"   卖出信号:")
        for signal, active in sell_signals.items():
            if signal != 'overall_sell' and active:
                print(f"     ⚠️ {signal}")
        
        overall_sell = "🔴 建议卖出" if sell_signals['overall_sell'] else "🟢 可以持有"
        print(f"   综合建议: {overall_sell}")

def demo_portfolio_management():
    """演示投资组合管理"""
    print("\n" + "="*60)
    print("💼 投资组合管理演示")
    print("="*60)
    
    from new_strategy import PortfolioManager
    
    # 创建投资组合管理器
    portfolio = PortfolioManager(initial_capital=1000000)
    
    print(f"初始资金: ¥{portfolio.initial_capital:,.2f}")
    
    # 模拟添加几个头寸
    positions_to_add = [
        {'code': '600519', 'name': '贵州茅台', 'price': 1680.0},
        {'code': '000858', 'name': '五粮液', 'price': 180.5},
        {'code': '300750', 'name': '宁德时代', 'price': 185.5},
        {'code': '000001', 'name': '平安银行', 'price': 10.5}
    ]
    
    print(f"\n📈 添加头寸:")
    for pos in positions_to_add:
        shares = portfolio.calculate_position_size(pos['price'])
        if shares > 0:
            portfolio.add_position(pos['code'], pos['name'], pos['price'], shares)
            cost = shares * pos['price']
            print(f"   ✓ {pos['name']}: {shares}股 @ ¥{pos['price']:.2f} = ¥{cost:,.2f}")
    
    # 模拟价格变化
    print(f"\n📊 价格更新:")
    price_changes = {
        '600519': 1720.0,  # 茅台上涨
        '000858': 175.2,   # 五粮液下跌
        '300750': 195.8,   # 宁德时代上涨
        '000001': 10.2     # 平安银行小跌
    }
    
    portfolio.update_positions(price_changes)
    for code, new_price in price_changes.items():
        if code in portfolio.positions:
            old_price = portfolio.positions[code]['entry_price']
            change_pct = (new_price - old_price) / old_price
            change_symbol = "📈" if change_pct > 0 else "📉"
            print(f"   {change_symbol} {portfolio.positions[code]['name']}: "
                  f"¥{old_price:.2f} → ¥{new_price:.2f} ({change_pct:+.1%})")
    
    # 显示投资组合摘要
    summary = portfolio.get_portfolio_summary()
    print(f"\n💰 投资组合摘要:")
    print(f"   当前现金: ¥{summary['current_cash']:,.2f}")
    print(f"   总资产价值: ¥{summary['total_value']:,.2f}")
    print(f"   总盈亏: ¥{summary['total_profit_loss']:,.2f}")
    print(f"   收益率: {summary['return_rate']:+.2%}")
    print(f"   持仓数量: {summary['positions_count']}")
    
    # 显示详细持仓
    print(f"\n📋 详细持仓:")
    print(f"{'股票':<8} {'数量':<8} {'成本价':<10} {'现价':<10} {'盈亏':<12} {'盈亏率':<8}")
    print("-" * 65)
    for code, pos in summary['positions'].items():
        profit_loss = (pos['current_price'] - pos['entry_price']) * pos['shares']
        profit_rate = (pos['current_price'] - pos['entry_price']) / pos['entry_price']
        print(f"{pos['name']:<8} {pos['shares']:<8} ¥{pos['entry_price']:<9.2f} "
              f"¥{pos['current_price']:<9.2f} ¥{profit_loss:<11.2f} {profit_rate:<+7.1%}")

def main():
    """主演示函数"""
    print("🚀 增强股票投资策略系统演示")
    print("=" * 80)
    print("本演示将展示系统的主要功能模块")
    print("注意：演示使用模拟数据，实际使用时会调用真实的股票数据API")
    print("=" * 80)
    
    try:
        # 1. 配置系统演示
        demo_config_system()
        
        # 2. 股票筛选演示
        filtered_stocks = demo_mock_screening()
        
        # 3. 评分系统演示
        demo_scoring_system()
        
        # 4. 交易信号演示
        demo_trading_signals()
        
        # 5. 投资组合管理演示
        demo_portfolio_management()
        
        print("\n" + "="*80)
        print("🎉 演示完成！")
        print("="*80)
        print("📚 更多信息请参考:")
        print("   - README.md: 详细使用说明")
        print("   - test_strategy.py: 运行完整测试")
        print("   - new_strategy.py: 查看完整实现")
        print("   - strategy_config.py: 配置系统详情")
        print()
        print("⚠️  免责声明:")
        print("   本系统仅供学习研究使用，不构成投资建议")
        print("   投资有风险，决策需谨慎")
        print("="*80)
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f"\n❌ 演示失败: {e}")
        print("请检查系统环境和依赖库是否正确安装")

if __name__ == "__main__":
    main()