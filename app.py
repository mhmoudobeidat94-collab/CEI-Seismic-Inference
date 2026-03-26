
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="CEI Earthquake System", layout="wide", page_icon="🚀")

st.title("🛰️ نظام مؤشر التأثير السببي (CEI)")
st.subheader("تحسين دقة إنذارات الزلازل — بحث علمي: محمود ماري عبيدات")
st.markdown("---")

@st.cache_data(ttl=3600)
def load_live_data():
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson"
        res = requests.get(url).json()
        quakes = [{
            'الموقع': f['properties']['place'],
            'القوة': f['properties']['mag'],
            'العمق': f['geometry']['coordinates'][2],
            'التوقيت': pd.to_datetime(f['properties']['time'], unit='ms')
        } for f in res['features']]
        return pd.DataFrame(quakes).dropna()
    except:
        return pd.DataFrame()

df = load_live_data()

st.sidebar.header("⚙️ إعدادات النموذج")
alpha = st.sidebar.slider("معامل ألفا (α) - حساسية العمق", 0.1, 5.0, 0.6)
threshold = st.sidebar.slider("عتبة القرار (Threshold)", 0.01, 0.2, 0.06)

if not df.empty:
    df['CEI_Score'] = df['القوة'] / (df['القوة'] + alpha * df['العمق'])
    df['CEI_Decision'] = df['CEI_Score'] >= threshold
    df['J_Alert'] = df['القوة'] >= 5.0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("إجمالي الأحداث", len(df))
    col2.metric("إنذارات J-ALERT", int(df['J_Alert'].sum()), delta="تقليدي", delta_color="inverse")
    prevented = df[(df['J_Alert'] == True) & (df['CEI_Decision'] == False)]
    col3.metric("إنذارات CEI الذكية", int(df['CEI_Decision'].sum()), delta=f"منع {len(prevented)} خطأ", delta_color="normal")

    fig = px.scatter(df, x="القوة", y="العمق", color="CEI_Decision", 
                     size=df['CEI_Score'].abs()*10 + 2, hover_name="الموقع",
                     color_discrete_map={True: "#ef4444", False: "#10b981"},
                     template="plotly_dark")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📝 سجل الإنذارات الكاذبة التي تم منعها")
    st.dataframe(prevented[['الموقع', 'القوة', 'العمق', 'CEI_Score']], use_container_width=True)
else:
    st.error("فشل الاتصال بالبيانات.")
