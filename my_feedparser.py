import feedparser
from datetime import datetime, timedelta
import time
import json
import os

url1 = "https://rsshub.app/gov/nea/sjzz/ghs"
url2 = "https://rsshub.app/eastmoney/report/strategyreport"
rsshub_url = url2

agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"

CACHE_FILE = "strategy_reports_cache.json"

def load_cache():
    """加载缓存数据"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                cache_date = datetime.strptime(cache_data['date'], '%Y-%m-%d').date()
                today = datetime.now().date()
                
                if cache_date == today:
                    print("使用缓存数据（今日数据）")
                    return cache_data['data']
                else:
                    print(f"缓存数据过期（缓存日期: {cache_date}，今日: {today}）")
                    return None
        except (json.JSONDecodeError, KeyError, ValueError):
            print("缓存文件格式错误，将重新获取数据")
            return None
    return None

def save_cache(data):
    """保存数据到缓存"""
    cache_data = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'data': data
    }
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    print(f"数据已缓存到 {CACHE_FILE}")

def get_strategy_reports():
    """从RSS源获取策略报告，只返回最近一周内的数据"""
    feed = load_cache()
    
    if not feed:
        # 如果 feed 是 None, 或者是一个空列表/字典，条件都会成立
        print("缓存为空或无效，正在从网络获取策略报告...")
        feed = feedparser.parse(url2, agent=agent)
    else:
        return feed
    print("正在获取策略报告...")
    
    if not feed.entries:
        return []
    
    # 计算一周前的时间
    one_week_ago = datetime.now() - timedelta(days=7)
    
    filtered_reports = []
    for entry in feed.entries:
        # 检查是否有发布日期
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            # 将发布时间转换为datetime对象
            pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            # 只添加最近一周内的报告
            if pub_date >= one_week_ago:
                filtered_reports.append({
                    "title": entry.title, 
                    "link": entry.link, 
                    "description": entry.description
                })
    
    print(f"成功获取到 {len(filtered_reports)} 篇最近7天内的策略报告。")
    
    # 保存到缓存
    save_cache(filtered_reports)
    
    return filtered_reports

if __name__ == "__main__":
    get_strategy_reports()