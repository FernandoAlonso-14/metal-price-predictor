# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predictor import MetalPredictor
from data_adapter import RealtimeDataManager

# 設定頁面配置
st.set_page_config(page_title="AI 金屬價格估價系統", layout="wide")

# 1. 載入模型 (使用 cache 避免每次操作都重跑)
@st.cache_resource
def load_predictor():
    return MetalPredictor()

predictor = load_predictor()
assets_list = predictor.get_asset_names()

# 2. 標題與側邊欄
st.title("🤖 AI 金屬原物料估價系統")
st.markdown("### 基於 Graph Transformer 的即時價格預測")

with st.sidebar:
    st.header("控制面板")
    selected_asset = st.selectbox("選擇要估價的金屬/資產", assets_list)
    refresh_btn = st.button("更新即時數據並預測")

# 3. 主邏輯
if refresh_btn or 'prediction_done' not in st.session_state:
    with st.spinner('正在從國際市場獲取最新數據並進行 AI 運算...'):
        # 初始化數據適配器
        dm = RealtimeDataManager(assets_list)
        
        # 獲取數據
        input_tensor = dm.get_live_data()
        indi_tensor = dm.get_indicator_data()
        
        # 預測
        prices, trends = predictor.predict(input_tensor, indi_tensor)
        
        # 將結果存入 session state
        st.session_state['prices'] = prices
        st.session_state['trends'] = trends
        st.session_state['prediction_done'] = True
        st.session_state['last_input'] = input_tensor # 保存輸入以顯示歷史數據

# 4. 顯示結果
if st.session_state.get('prediction_done'):
    prices = st.session_state['prices'] # shape (21, 4)
    trends = st.session_state['trends'] # shape (21, 4)
    
    # 找到選定資產的 index
    asset_idx = assets_list.index(selected_asset)
    
    # 獲取該資產的預測值
    pred_prices = prices[asset_idx] # 未來4週的價格
    pred_probs = trends[asset_idx]  # 未來4週的上漲機率
    
    # 顯示關鍵指標
    col1, col2, col3 = st.columns(3)
    
    current_price = st.session_state['last_input'][0, asset_idx, -1, -1, 3].item() # 取得最近一天的 Close (假設 index 3 是 Close)
    
    with col1:
        st.metric("當前參考價格", f"{current_price:.2f}")
    
    with col2:
        next_week_price = pred_prices[0]
        delta = next_week_price - current_price
        st.metric("下週預測均價", f"{next_week_price:.2f}", f"{delta:.2f}")
        
    with col3:
        confidence = pred_probs[0] * 100
        trend_text = "看漲 📈" if confidence > 50 else "看跌 📉"
        st.metric("AI 趨勢判斷", trend_text, f"信心度 {confidence:.1f}%")

    # 繪圖
    st.subheader(f"{selected_asset} - 未來 4 週價格趨勢預測")
    
    # 製作圖表數據
    weeks = [f'Week {i+1}' for i in range(4)]
    chart_data = pd.DataFrame({
        '預測價格': pred_prices
    }, index=weeks)
    
    st.line_chart(chart_data)
    
    # 表格詳情
    st.subheader("詳細預測數據")
    df_detail = pd.DataFrame({
        '週次': weeks,
        '預測均價': pred_prices,
        '上漲機率': [f"{p*100:.1f}%" for p in pred_probs]
    })
    st.table(df_detail)

else:
    st.info("請點擊側邊欄的按鈕開始預測")