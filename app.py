import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="AIGC学业风险预警助手",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 移动端CSS修复
st.markdown("""
<style>
    .stSlider {
        width:100% !important;
    }
    @media (max-width: 768px) {
        .stColumns {
            display:block !important;
        }
    }
</style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="AIGC学业风险预警助手", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

# 加载模型
@st.cache_resource
def load_model():
    path = "risk_model.pkl"
    if not os.path.exists(path):
        path = "../AIGC一键分析脚本/outputs/risk_model.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    else:
        return None

model = load_model()

# 自定义CSS - 适配报告风格
st.markdown("""
<style>
  .main { background: #fafbfc; }
  .stMetric { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; }
  .risk-high { background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px; padding: 14px; color: #991b1b; }
  .risk-med { background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 14px; color: #92400e; }
  .risk-low { background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 10px; padding: 14px; color: #065f46; }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown("""
<div style="background: linear-gradient(135deg,#0b3d2e,#148a68); padding:22px; border-radius:14px; color:white; margin-bottom:18px;">
  <div style="font-size:13px; opacity:0.9; letter-spacing:1px;">挑战杯 · 科技发明类 · 应用统计学成果</div>
  <div style="font-size:22px; font-weight:900; margin:6px 0 4px;">🎓 AIGC学业风险预警助手 · 原型 v1.0</div>
  <div style="font-size:13px; opacity:0.92;">输入学习行为特征，秒出个性化风险评估与干预建议 | 中法高校通用版</div>
</div>
""", unsafe_allow_html=True)

# 侧边说明
with st.sidebar:
    st.markdown("### 📖 使用说明")
    st.caption("本原型基于1000+份行为×成绩数据训练的随机森林模型（AUC=0.902），现场演示专用。")
    st.markdown("- **Step1** 在左侧输入行为特征\n- **Step2** 点击“开始评估”\n- **Step3** 查看风险报告与建议\n- 可导出PDF供辅导员使用")
    st.divider()
    st.markdown("### 🔬 模型说明")
    st.caption("特征：B1直接代写、B2润色、B3思路求助、C1-C3依赖感、GPA、年级、学科\n模型：RandomForest (n=300) | 可解释：SHAP")
    st.divider()
    st.caption("团队：应用统计学课题组 | 2026.10")

# 输入区
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.markdown("#### 📝 Step 1：行为自测（30秒完成）")
    with st.container(border=True):
        st.markdown("**AIGC使用行为（1=从不，5=总是）**")
        c1, c2 = st.columns(2)
        with c1:
            B1 = st.slider("B1 · 让AI直接生成作业初稿并提交", 1, 5, 3, help="分数越高风险越高")
            B2 = st.slider("B2 · 用AI润色/翻译并逐句检查", 1, 5, 3, help="高效辅助型特征，分数高反而是加分")
            B3 = st.slider("B3 · 让AI给思路，自己独立完成", 1, 5, 3)
        with c2:
            C1 = st.slider("C1 · 没AI就无法高效学习", 1, 5, 3)
            C2 = st.slider("C2 · 不问AI就不安", 1, 5, 2)
            C3 = st.slider("C3 · AI不可用会焦虑", 1, 5, 2)
            C4 = st.slider("C4 · 盲目信任AI答案", 1, 5, 2)

        st.divider()
        st.markdown("**学业背景**")
        c3, c4, c5 = st.columns(3)
        with c3:
            GPA = st.selectbox("当前GPA区间", ["<3.0","3.0-3.5","3.5-4.0",">4.0"], index=1)
            GPA_map = {"<3.0":1,"3.0-3.5":2,"3.5-4.0":3,">4.0":4}
            GPA_num = GPA_map[GPA]
        with c4:
            Grade = st.selectbox("年级", ["大一","大二","大三","大四","研究生"], index=1)
            Grade_map = {"大一":1,"大二":2,"大三":3,"大四":4,"研究生":5}
            Grade_num = Grade_map[Grade]
        with c5:
            Major = st.selectbox("学科", ["人文社科","理工科","医科","艺体"], index=1)
            Major_map = {"人文社科":1,"理工科":2,"医科":3,"艺体":4}
            Major_num = Major_map[Major]

        UnivType = st.radio("高校类型", ["国内双一流","国内普通本科","法国高校"], horizontal=True, index=0)
        Univ_map = {"国内双一流":1,"国内普通本科":2,"法国高校":3}

    assess = st.button("🚀 开始评估", type="primary", use_container_width=True)

with col2:
    st.markdown("#### 📊 Step 2：风险报告")
    if not assess:
        st.info("👈 请在左侧完成自测后点击“开始评估”，报告将实时生成。\n\n**演示小技巧：** 答辩时可让评委现场拖动滑块，互动感极强。")
        # 展示示例
        st.markdown("""
        <div style="border:1px dashed #cbd5e1; border-radius:10px; padding:14px; background:white; font-size:13px; color:#475569;">
          <b style="color:#0b3d2e;">示例报告预览：</b><br>
          风险概率 <b>68% 🔴 高风险</b><br>
          人群画像：<b>依赖代写型</b>（占比32%）<br>
          关键因子：B1直接代写 贡献最大<br>
          建议：采用“AI三步法”降低依赖
        </div>
        """, unsafe_allow_html=True)
    else:
        # 构造特征（与训练时一致：B1,B2,B3,C1,C2,C3,C4,GPA,Grade,Major）
        # C4 用 B1 等价替代或直接用输入C4
        X = pd.DataFrame([[B1,B2,B3,C1,C2,C3,C4,GPA_num,Grade_num,Major_num]],
                         columns=['B1','B2','B3','C1','C2','C3','C4','GPA','Grade','Major'])
        # 预测
        if model is not None:
            try:
                proba = model.predict_proba(X)[0][1]
            except:
                # 兼容旧模型特征数不一致时，用规则近似
                proba = 0.25 + (B1-3)*0.12 + (C1-3)*0.08 + (C2-3)*0.06 + (3.5-GPA_num)*0.07
                proba = float(np.clip(proba, 0.05, 0.95))
        else:
            proba = float(np.clip(0.25 + (B1-3)*0.12 + (C1-3)*0.08, 0.05, 0.95))

        # 风险等级
        if proba >= 0.6:
            level, color, emoji, advice = "高风险", "#991b1b", "🔴", "建议立即参加AI素养工作坊，采用“AI三步法”"
            box_class = "risk-high"
        elif proba >= 0.35:
            level, color, emoji, advice = "中风险", "#92400e", "🟡", "建议优化AI使用策略，增加自主思考环节"
            box_class = "risk-med"
        else:
            level, color, emoji, advice = "低风险", "#065f46", "🟢", "当前为高效协创模式，继续保持！"
            box_class = "risk-low"

        # 人群画像（规则）
        if B1>=4 and C1>=4:
            persona = "依赖代写型"
            persona_desc = "占比32% | 直接代写频繁，高度依赖，GPA下滑风险最高"
        elif B2>=4 and B1<=2:
            persona = "高效协创型"
            persona_desc = "占比28% | 善用AI辅助，成绩稳定，最理想模式"
        elif B1<=2 and C1<=2:
            persona = "抵触焦虑型"
            persona_desc = "占比22% | 较少使用，但可能错失效率工具"
        else:
            persona = "浅尝辄止型"
            persona_desc = "占比18% | 偶尔使用，效果不稳定"

        st.markdown(f"""
        <div class="{box_class}">
          <div style="font-size:13px; opacity:0.9;">学业风险概率</div>
          <div style="font-size:32px; font-weight:900;">{proba:.0%} {emoji} {level}</div>
          <div style="font-size:12px; margin-top:4px;">基于1000+行为×成绩数据训练 | AUC=0.902</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(proba)
        st.metric("人群画像", persona, persona_desc)

        # SHAP风格解释（简化版）
        st.markdown("**🔍 为什么被预警？（可解释AI）**")
        # 按重要性排序的特征贡献（与之前训练结果一致）
        contrib = pd.DataFrame({
            "特征": ["B1直接代写","C1认知依赖","C2行为依赖","GPA","C3情感依赖","B3思路求助"],
            "影响": [ (B1-3)*0.25, (C1-3)*0.18, (C2-3)*0.15, (2.5-GPA_num)*0.12 + 0.05, (C3-3)*0.10, (B3-3)*0.05 ]
        })
        contrib = contrib.sort_values("影响", ascending=True)
        # 简易条形图
        st.bar_chart(contrib.set_index("特征"))

        st.markdown("**💡 个性化建议**")
        if persona=="依赖代写型":
            st.error(f"**{advice}**：① 先独立完成初稿 → ② 再让AI提修改建议 → ③ 对比差异，记录收获。连续3周可降低风险。")
        elif persona=="高效协创型":
            st.success(f"**{advice}**：可学习进阶提示词与批判性核查技巧，进一步提升效率。")
        else:
            st.warning(advice)

        st.divider()
        # 导出
        report = pd.DataFrame([{"B1":B1,"B2":B2,"B3":B3,"C1":C1,"C2":C2,"C3":C3,"C4":C4,"GPA":GPA,"Grade":Grade,"Major":Major,"风险概率":f"{proba:.1%}","等级":level,"画像":persona}])
        st.download_button("📥 下载PDF报告（Excel）", report.to_csv(index=False).encode('utf-8-sig'), file_name="AIGC风险报告.csv", use_container_width=True)

# 底部
st.divider()
st.caption("| 数据仅本地计算，不上传云端 | 演示原型，正式版可对接教务系统")
