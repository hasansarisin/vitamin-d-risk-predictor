import streamlit as st
import pandas as pd, joblib, json
from pathlib import Path

st.set_page_config(page_title="NHANES Vitamin D Risk", page_icon="☀️", layout="centered")
ROOT=Path(__file__).parent
model=joblib.load(ROOT/'vitd_xgb_model.joblib')
metrics=json.load(open(ROOT/'metrics.json'))

st.title("☀️ Vitamin D Deficiency Risk — Research Prototype")
st.caption("NHANES August 2021–August 2023 verileriyle eğitilmiş araştırma amaçlı tahmin aracı")
st.warning("Bu araç tanı koymaz ve vitamin D takviyesi önermez. Sonuç yalnızca araştırma/tarama amaçlı model tahminidir. Klinik karar için hekim değerlendirmesi ve gerektiğinde serum 25(OH)D ölçümü gerekir.")

with st.form('risk'):
    c1,c2=st.columns(2)
    with c1:
        age=st.number_input('Yaş',min_value=2,max_value=80,value=40,step=1)
        sex_label=st.selectbox('Cinsiyet',['Erkek','Kadın'])
        bmi=st.number_input('BMI (kg/m²)',min_value=10.0,max_value=80.0,value=25.0,step=.1)
        diabetes_label=st.selectbox('Doktor tarafından diyabet tanısı',['Hayır','Evet'])
    with c2:
        race_label=st.selectbox('NHANES ırk/etnik grup kategorisi',[
            'Mexican American','Other Hispanic','Non-Hispanic White','Non-Hispanic Black','Non-Hispanic Asian','Other / Multiracial'])
        hh=st.number_input('Hane büyüklüğü',min_value=1,max_value=7,value=2,step=1)
        pir=st.number_input('Household PIR',min_value=0.0,max_value=5.0,value=2.0,step=.1,help='Gelirin federal yoksulluk eşiğine oranı; NHANES değişkeni INDFMPIR.')
    submitted=st.form_submit_button('Riski Hesapla',use_container_width=True)

if submitted:
    sex={'Erkek':1,'Kadın':2}[sex_label]
    race={'Mexican American':1,'Other Hispanic':2,'Non-Hispanic White':3,'Non-Hispanic Black':4,'Non-Hispanic Asian':6,'Other / Multiracial':7}[race_label]
    diabetes={'Hayır':0,'Evet':1}[diabetes_label]
    row=pd.DataFrame([{'RIAGENDR':sex,'RIDAGEYR':age,'RIDRETH3':race,'DMDHHSIZ':hh,'INDFMPIR':pir,'BMXBMI':bmi,'Diabetes':diabetes}])
    prob=float(model.predict_proba(row)[0,1])
    st.metric('Modelin tahmin ettiği D vitamini eksikliği olasılığı',f'{prob*100:.1f}%')
    if prob < .20: label='Daha düşük model tahmini'
    elif prob < .40: label='Orta model tahmini'
    else: label='Daha yüksek model tahmini'
    st.info(label + ' — Bu bantlar klinik eşik değildir; yalnızca tahminin okunmasını kolaylaştırır.')
    st.progress(min(max(prob,0.0),1.0))

st.divider()
st.subheader('Model hakkında')
st.write(f"Model: XGBoost • Uygun örneklem: {metrics['n_total']:,} kişi • Test AUC: {metrics['auc']:.3f} • Brier: {metrics['brier']:.3f}")
st.write("Hedef: serum total 25(OH)D < 50 nmol/L. Bu eşik, karşılaştırılan 2024 çalışmasının araştırma tanımını yeniden üretmek amacıyla kullanılmıştır.")
st.caption("Süt tüketimi 2021–2023 DBQ_L içinde makaledeki değişkenle birebir eşleşmediği için eklenmedi. İlk oluşturulan household-smoking alanı da varyasyon göstermediğinden bu sürümden çıkarıldı. Böylece arayüzde sahte/uydurulmuş girdi kullanılmadı.")
