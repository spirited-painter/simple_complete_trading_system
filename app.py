import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates

# --- 页面配置 ---
st.set_page_config(page_title="量化定投策略回测", layout="wide")

# --- 常量定义 ---
INDEX_MAP = {
    "创业板指": "sz399006", "沪深300": "sh000300", "上证50": "sh000016",
    "中证500": "sh000905", "中证1000": "sh000852", "科创50": "sh000688",
    "上证综合指数": "sh000001", "中证银行": "sz399986", "中证券商": "sz399975",
    "中证保险": "sz399809", "中证主要消费": "sh000932", "中证可选消费": "sh000931",
    "国证食品饮料": "sz399396", "中证白酒": "sz399997", "中证医药卫生": "sh000933",
    "中证房地产": "sh000952", "中证基建工程": "sz399995", "中证能源": "sh000928",
    "中证材料": "sh000929",
}

# 默认规则 (转换为 DataFrame 以便在 data_editor 中使用)
DEFAULT_BUY_RULES = pd.DataFrame([
    {"阈值(%)": -35.0, "比例(%)": 100.0},
    {"阈值(%)": -30.0, "比例(%)": 17.6},
    {"阈值(%)": -25.0, "比例(%)": 10.5},
    {"阈值(%)": -20.0, "比例(%)": 5.0}
])

DEFAULT_SELL_RULES = pd.DataFrame([
    {"阈值(%)": 40.0, "比例(%)": 100.0},
    {"阈值(%)": 35.0, "比例(%)": 17.6},
    {"阈值(%)": 30.0, "比例(%)": 10.5},
    {"阈值(%)": 25.0, "比例(%)": 5.0},
    {"阈值(%)": 20.0, "比例(%)": 0.0}
])

# --- 核心计算函数 ---
def fetch_and_process_data(index_code):
    """获取数据并进行月线重采样"""
    try:
        daily_data = ak.stock_zh_index_daily(symbol=index_code)
        if daily_data.empty:
            return None
    except Exception as e:
        st.error(f"获取数据出错: {e}")
        return None

    daily_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data.set_index('date', inplace=True)
    
    # Resample to Month End
    monthly_data = daily_data.resample('ME').apply({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    monthly_data.dropna(inplace=True)
    monthly_data.index.name = '日期' # 保持为 datetime索引以便后续处理
    
    processed_data = monthly_data[['Close']].copy()
    processed_data.rename(columns={'Close': '当月收盘价'}, inplace=True)
    # 插入序号
    processed_data.reset_index(inplace=True)
    processed_data.insert(1, '序号', range(1, len(processed_data) + 1))
    
    return processed_data

def perform_backtest(processed_data, slope, intercept, buy_rules_list, sell_rules_list, backtest_start_date, extra_initial_cash):
    """执行回测逻辑"""
    # 计算理论值和百分比
    processed_data['理论值'] = slope * processed_data['序号'] + intercept
    processed_data['百分比'] = (processed_data['当月收盘价'] - processed_data['理论值']) / processed_data['理论值']
    
    # 准备回测变量
    results_lists = {k: [] for k in ['shares', 'cash', 'stock_value', 'total_assets', 'cumulative_investment', 'profit', 'actual_monthly_return', 'annual_irr', 'max_drawdown', 'volatility', 'net_value_index']}
    shares_held = 0.0
    cash_held = 0.0
    cumulative_investment = 0.0
    net_value_index = 1000.0
    peak_net_value_index = 1000.0
    actual_monthly_returns_history = []
    trading_started = False
    
    # 转换日期格式用于比较
    start_date_ts = pd.Timestamp(backtest_start_date)

    for i, row in processed_data.iterrows():
        current_date = row['日期']
        
        # 1. 检查是否到达回测开始时间
        if current_date < start_date_ts:
            # 填充空值或0
            for key in results_lists:
                val = np.nan if key == 'annual_irr' else (1000.0 if key in ['net_value_index'] else 0.0)
                results_lists[key].append(val)
            actual_monthly_returns_history.append(0.0)
            continue
            
        # 2. 初始化交易状态 (首月)
        if not trading_started:
            trading_started = True
            cash_held = 1.0 + extra_initial_cash
            cumulative_investment = 1.0 + extra_initial_cash
            net_value_index = 1000.0
            peak_net_value_index = 1000.0
            actual_monthly_returns_history = []
        else:
            # 后续月份定投
            cash_held += 1.0
            cumulative_investment += 1.0
            
        close_price = row['当月收盘价']
        # 安全获取上月资产
        last_month_assets = results_lists['total_assets'][-1] if results_lists['total_assets'] else 0.0
        
        # 3. 交易决策
        percentage = row['百分比']
        
        # 处理规则为空的情况
        buy_trigger = max([r[0] for r in buy_rules_list] + [-999])
        sell_trigger = min([r[0] for r in sell_rules_list] + [999])
        
        if percentage <= buy_trigger:
            # 买入：按阈值从小到大排序（越低估越优先）
            for threshold, ratio in sorted(buy_rules_list, key=lambda item: item[0]):
                if percentage <= threshold:
                    cash_to_spend = cash_held * ratio
                    shares_bought = cash_to_spend / close_price
                    shares_held += shares_bought
                    cash_held -= cash_to_spend
                    break
        elif percentage >= sell_trigger:
            # 卖出：按阈值从大到小排序（越高估越优先）
            for threshold, ratio in sorted(sell_rules_list, key=lambda item: item[0], reverse=True):
                if percentage >= threshold:
                    shares_to_sell = shares_held * ratio
                    cash_gained = shares_to_sell * close_price
                    shares_held -= shares_to_sell
                    cash_held += cash_gained
                    break
                    
        # 4. 结算与指标计算
        stock_value = shares_held * close_price
        total_assets = stock_value + cash_held
        profit = total_assets - cumulative_investment
        
        # 当月真实收益率计算
        # 如果是刚开始交易的第一个月(i对应start_date_ts)，基数需要包含extra_cash
        # 这里逻辑简化：如果是trading_started且不是第一天，用上月资产+1做分母
        # 如果是第一天，收益率暂记为0
        if i == processed_data[processed_data['日期'] >= start_date_ts].index[0]:
             capital_base = 0 # 第一月无法计算相对于上月的收益
             actual_monthly_return = 0.0
        else:
             capital_base = last_month_assets + 1.0
             actual_monthly_return = (total_assets - capital_base) / capital_base if capital_base > 0 else 0.0
             
        actual_monthly_returns_history.append(actual_monthly_return)
        net_value_index *= (1 + actual_monthly_return)
        
        # 波动率
        volatility = np.std([r for r in actual_monthly_returns_history if r != 0], ddof=1) if len([r for r in actual_monthly_returns_history if r != 0]) > 1 else 0.0
        
        # 最大回撤
        peak_net_value_index = max(peak_net_value_index, net_value_index)
        drawdown = (net_value_index - peak_net_value_index) / peak_net_value_index if peak_net_value_index != 0 else 0.0
        # 获取历史最大回撤
        prev_max_drawdown = min([r for r in results_lists['max_drawdown'] if r < 0] + [0.0])
        max_drawdown = min(prev_max_drawdown, drawdown)
        
        # IRR (仅当数据量足够时计算)
        annual_irr = np.nan
        # 找到开始回测的索引
        start_idx = processed_data[processed_data['日期'] >= start_date_ts].index[0]
        if (i - start_idx) >= 11:
            num_periods = i - start_idx + 1
            cash_flows = [-1.0] * num_periods
            cash_flows[0] -= extra_initial_cash # 首期流出增加额外现金
            cash_flows[-1] += total_assets # 末期流入
            try:
                monthly_irr = npf.irr(cash_flows)
                if not np.isnan(monthly_irr):
                    annual_irr = (1 + monthly_irr)**12 - 1
            except:
                pass
                
        # 存入结果
        current_vals = [shares_held, cash_held, stock_value, total_assets, cumulative_investment, profit, actual_monthly_return, annual_irr, max_drawdown, volatility, net_value_index]
        for key, val in zip(results_lists.keys(), current_vals):
            results_lists[key].append(val)
            
    # 将结果合并回 DataFrame
    # 映射列名
    col_map = {
        'shares': '持有股票数量', 'cash': '现金', 'stock_value': '股票价值', 
        'total_assets': '总资产', 'cumulative_investment': '累计投资', 'profit': '收益',
        'actual_monthly_return': '当月真实收益率', 'annual_irr': '年化收益率(IRR)',
        'max_drawdown': '最大回撤', 'volatility': '历史波动率', 'net_value_index': '净值指数'
    }
    for key, col_name in col_map.items():
        processed_data[col_name] = results_lists[key]
        
    # 计算仓位百分比
    processed_data['仓位百分比'] = np.where(processed_data['总资产'] > 0, 1 - (processed_data['现金'] / processed_data['总资产']), np.nan)
    
    return processed_data

# --- 绘图函数 ---
def plot_results(data, index_code, slope, intercept, start_date):
    # 准备绘图数据 (过滤掉回测开始前的数据用于绘图，或者全部显示但标记开始点)
    # 为了清晰，我们只绘制回测开始后的部分，或者全量显示但重点在回测期
    # 这里选择全量显示，但计算重合点
    
    dates_for_plot = data['日期']
    start_date_ts = pd.Timestamp(start_date)
    
    # --- 图1: 指数 vs 趋势 & 仓位 ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(f'{index_code} vs. Linear Trend & Position %', fontsize=14)
    ax1.plot(dates_for_plot, data['当月收盘价'], label=f'{index_code} (Close)', color='blue', linewidth=1.5)
    ax1.plot(dates_for_plot, data['理论值'], label='Trendline', color='red', linestyle='--', linewidth=1.5)
    ax1.set_ylabel('Points', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2_pos = ax1.twinx()
    ax2_pos.plot(dates_for_plot, data['仓位百分比'], label='Position %', color='purple', linestyle=':', alpha=0.6)
    ax2_pos.set_ylabel('Position %', color='purple')
    ax2_pos.tick_params(axis='y', labelcolor='purple')
    ax2_pos.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # 统一图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2_pos.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # X轴格式化
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    fig1.autofmt_xdate()
    st.pyplot(fig1)
    
    # --- 图2: 累计收益 ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(dates_for_plot, data['收益'], label='Cumulative P/L', color='purple', linewidth=1.5)
    ax2.fill_between(dates_for_plot, data['收益'], where=(data['收益'] >= 0), color='mediumpurple', alpha=0.3)
    ax2.fill_between(dates_for_plot, data['收益'], where=(data['收益'] < 0), color='lightcoral', alpha=0.3)
    ax2.set_title('Cumulative Profit/Loss', fontsize=14)
    ax2.set_ylabel('Profit (CNY)')
    ax2.legend(loc='upper left')
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator())
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    fig2.autofmt_xdate()
    st.pyplot(fig2)
    
    # --- 图3: 业绩对比 (基准化) ---
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax4 = ax3.twinx()
    
    # 找到回测开始的数据行
    mask = data['日期'] >= start_date_ts
    if not mask.any():
        st.warning("选定的回测开始日期超出数据范围。")
        return

    start_idx = data[mask].index[0]
    
    # 获取切片
    y1_slice = data['当月收盘价'].iloc[start_idx:]
    y2_slice = data['净值指数'].iloc[start_idx:]
    
    if len(y1_slice) > 0:
        base_price = y1_slice.iloc[0]
        base_net_value = y2_slice.iloc[0]
        
        # 计算百分比 (Rebase)
        # 注意：只显示回测开始后的部分，或者全量显示但前面为NaN
        price_pct = (data['当月收盘价'] / base_price - 1)
        price_pct.loc[:start_idx-1] = np.nan # 隐藏开始前的
        
        net_pct = (data['净值指数'] / base_net_value - 1)
        net_pct.loc[:start_idx-1] = np.nan
        
        ax3.plot(dates_for_plot, price_pct, color='dodgerblue', label=f'{index_code} (Rebased)', linewidth=1.5)
        ax4.plot(dates_for_plot, net_pct, color='crimson', label='Strategy (Rebased)', linewidth=1.5)
        
        # 统一Y轴范围
        # 计算回测期间的全局最大最小值
        valid_y1 = price_pct.dropna()
        valid_y2 = net_pct.dropna()
        
        y_min = min(valid_y1.min(), valid_y2.min())
        y_max = max(valid_y1.max(), valid_y2.max())
        margin = (y_max - y_min) * 0.05
        
        ax3.set_ylim(y_min - margin, y_max + margin)
        ax4.set_ylim(y_min - margin, y_max + margin)
        
        ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax4.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        ax3.set_ylabel('Price Return (%)', color='dodgerblue')
        ax4.set_ylabel('Strategy Return (%)', color='crimson')
        ax3.tick_params(axis='y', labelcolor='dodgerblue')
        ax4.tick_params(axis='y', labelcolor='crimson')
        
        ax3.set_title(f'Performance Comparison (Rebased to {start_date})', fontsize=14)
        
        # 图例
        l1, lab1 = ax3.get_legend_handles_labels()
        l2, lab2 = ax4.get_legend_handles_labels()
        ax4.legend(l1 + l2, lab1 + lab2, loc='upper left')
        
        ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.xaxis.set_minor_locator(mdates.MonthLocator())
        ax3.grid(True, which='both', linestyle='--', alpha=0.5)
        fig3.autofmt_xdate()
        st.pyplot(fig3)

    # --- 图4: 偏差分布 ---
    fig4, ax4_hist = plt.subplots(figsize=(12, 6))
    ax4_hist.hist(data['百分比'], bins=50, edgecolor='black', alpha=0.75, color='skyblue')
    ax4_hist.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax4_hist.set_title('Deviation Distribution', fontsize=14)
    ax4_hist.set_xlabel('Deviation (%)')
    ax4_hist.set_ylabel('Frequency')
    ax4_hist.axvline(x=0, color='r', linestyle='--', label='0%')
    ax4_hist.legend()
    st.pyplot(fig4)


# --- Streamlit 主程序逻辑 ---

# 1. 侧边栏输入
with st.sidebar:
    st.header("1. 选择指数")
    index_key = st.selectbox("预设指数", list(INDEX_MAP.keys()))
    custom_input = st.text_input("或输入自定义代码 (如 sh000001)")
    
    target_code = custom_input.strip() if custom_input else INDEX_MAP[index_key]
    target_name = custom_input.strip() if custom_input else index_key
    
    if st.button("加载数据", type="primary"):
        st.session_state['data_loaded'] = False # Reset
        with st.spinner("正在获取数据..."):
            df = fetch_and_process_data(target_code)
            if df is not None:
                st.session_state['raw_data'] = df
                st.session_state['index_name'] = target_name
                st.session_state['index_code'] = target_code
                st.session_state['data_loaded'] = True
                st.success(f"成功加载 {target_name} 数据！")
            else:
                st.error("数据获取失败，请检查代码或网络。")

# 2. 主界面逻辑
if st.session_state.get('data_loaded', False):
    df = st.session_state['raw_data']
    name = st.session_state['index_name']
    code = st.session_state['index_code']
    
    st.header(f"分析控制面板: {name} ({code})")
    
    # --- 高级选项配置区 ---
    with st.expander("⚙️ 高级参数设置 (点击展开)", expanded=True):
        col1, col2 = st.columns(2)
        
        # 日期范围选择
        date_options = df['日期'].dt.date.unique()
        with col1:
            fit_start = st.selectbox("拟合开始日期", date_options, index=0)
            fit_end = st.selectbox("拟合结束日期", date_options, index=len(date_options)-1)
            
            # 拟合计算按钮
            if st.button("计算推荐斜率/截距"):
                mask = (df['日期'].dt.date >= fit_start) & (df['日期'].dt.date <= fit_end)
                slice_df = df[mask]
                if not slice_df.empty:
                    slope_cal, intercept_cal = np.polyfit(slice_df['序号'], slice_df['当月收盘价'], 1)
                    st.session_state['rec_slope'] = slope_cal
                    st.session_state['rec_intercept'] = intercept_cal
                    st.success(f"计算完成: 斜率={slope_cal:.4f}, 截距={intercept_cal:.4f}")
        
        with col2:
            # 斜率截距输入 (使用 session_state 填充推荐值)
            slope = st.number_input("斜率 (Slope)", value=st.session_state.get('rec_slope', 0.0), format="%.4f")
            intercept = st.number_input("截距 (Intercept)", value=st.session_state.get('rec_intercept', 0.0), format="%.4f")
            
        st.markdown("---")
        st.subheader("交易规则设置")
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**买入规则 (低估)**")
            # 使用 data_editor 编辑规则
            buy_df = st.data_editor(DEFAULT_BUY_RULES, num_rows="dynamic", key="buy_editor")
        
        with col4:
            st.markdown("**卖出规则 (高估)**")
            sell_df = st.data_editor(DEFAULT_SELL_RULES, num_rows="dynamic", key="sell_editor")
            
        st.markdown("---")
        col5, col6 = st.columns(2)
        with col5:
            backtest_start = st.selectbox("回测起始日期", date_options, index=0, key="bt_start")
        with col6:
            extra_cash = st.number_input("额外初始现金", value=0.0, min_value=0.0)

        run_btn = st.button("运行回测分析", type="primary")

    # --- 运行回测并展示结果 ---
    if run_btn:
        # 1. 如果斜率还是0，先自动计算全量的
        if slope == 0 and intercept == 0:
             s, i_val = np.polyfit(df['序号'], df['当月收盘价'], 1)
             slope = s
             intercept = i_val
             st.info(f"使用全量数据自动计算拟合参数: 斜率={slope:.4f}, 截距={intercept:.4f}")

        # 2. 解析规则
        # DataFrame -> List of tuples [(threshold, ratio), ...]
        # 注意：用户输入的是百分数 (e.g. 35)，代码逻辑需要小数 (0.35)
        # 修改：原代码逻辑里，DEFAULT_BUY_RULES里是 -0.35，但这里为了方便用户编辑，data_editor显示的是 -35.0
        # 所以转换时需要 / 100.0
        
        b_rules = []
        for _, row in buy_df.iterrows():
            if pd.notna(row['阈值(%)']) and pd.notna(row['比例(%)']):
                b_rules.append((row['阈值(%)']/100.0, row['比例(%)']/100.0))
        
        s_rules = []
        for _, row in sell_df.iterrows():
            if pd.notna(row['阈值(%)']) and pd.notna(row['比例(%)']):
                s_rules.append((row['阈值(%)']/100.0, row['比例(%)']/100.0))

        # 3. 执行计算
        final_df = perform_backtest(
            df.copy(), 
            slope, 
            intercept, 
            b_rules, 
            s_rules, 
            backtest_start, 
            extra_cash
        )
        
        # 4. 格式化表格用于展示
        st.subheader("📊 回测结果数据")
        # 将日期转为字符串以便展示
        display_df = final_df.copy()
        display_df['日期'] = display_df['日期'].dt.strftime('%Y-%m')
        
        # 设置 Pandas Styler
        # 注意：Streamlit 的 dataframe 支持 pandas styler
        st.dataframe(
            display_df.style.format({
                '当月收盘价': '{:.3f}', '理论值': '{:.2f}', '百分比': '{:.2%}', 
                '持有股票数量': '{:.4f}', '股票价值': '{:,.2f}', '现金': '{:,.2f}', 
                '仓位百分比': '{:.2%}', '总资产': '{:,.2f}', '累计投资': '{:.3f}', 
                '收益': '{:,.2f}', '当月真实收益率': '{:.2%}', '净值指数': '{:,.2f}', 
                '年化收益率(IRR)': '{:.2%}', '最大回撤': '{:.2%}', '历史波动率': '{:.2%}'
            }, na_rep='NA'),
            use_container_width=True
        )
        
        # 5. 绘制图表
        st.subheader("📈 可视化分析")
        plot_results(final_df, code, slope, intercept, backtest_start)

else:
    st.info("👈 请在左侧侧边栏选择指数并点击“加载数据”开始。")