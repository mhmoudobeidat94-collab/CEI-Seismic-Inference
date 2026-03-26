# CEI-Seismic-Inference
"A Causal Inference system to reduce false seismic alarms using Magnitude and Depth correlation. Built with Streamlit and USGS live data."
# 🛰️ Causal Effect Index (CEI) for Seismic Alarm Optimization
### بحث وتطوير: محمود عبيدات (Mahmoud Obeidat)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://cei-seismic-inference-8zpmbriyt4th3tpqhdy7rt.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 نبذة عن المشروع (Project Overview)
يعد نظام **CEI** حلاً ابتكارياً تم تطويره لتقليل "الإنذارات الكاذبة" في أنظمة رصد الزلازل التقليدية. يعتمد المشروع على مبادئ **الاستدلال السببي (Causal Inference)** لتمييز الهزات الأرضية ذات التأثير الحقيقي عن تلك التي تفتقر للكتلة أو العمق المؤثر، مما يساهم في تقليل التكلفة الاقتصادية والنفسية للتحذيرات الخاطئة.

## 🧠 الجانب العلمي (The Science)
بناءً على أبحاثي في **جامعة آل البيت**، قمت باختبار خوارزمية **CEI** التي تحلل العلاقة السببية بين:
1. **قوة الزلزال (Magnitude):** المقياس التقليدي للخطر.
2. **العمق (Depth):** المعامل الحاسم في تحديد مدى وصول الطاقة للسطح.
3. **معامل ألفا (α):** معامل مرن للتحكم في حساسية النظام تجاه العمق.

## 🛠️ الميزات التقنية (Technical Features)
* **بيانات لحظية (Live Data):** يتم سحب البيانات مباشرة من هيئة المساحة الجيولوجية الأمريكية (USGS).
* **لوحة تحكم تفاعلية (Interactive Dashboard):** تتيح للمستخدمين تعديل معاملات النموذج ورؤية النتائج فوراً باستخدام Streamlit.
* **تصور البيانات (Data Visualization):** استخدام مكتبة Plotly لعرض خرائط وتوزيعات إحصائية لأكثر من 10,000 حدث زلزالي شهرياً.
* **فلترة ذكية (Smart Filtering):** سجل خاص يعرض بدقة عدد الإنذارات التي نجح نظام CEI في "منعها" مقارنة بالنظام التقليدي J-ALERT.

## 💻 الأدوات المستخدمة (Tech Stack)
* **Language:** Python 3.x
* **Framework:** Streamlit
* **Libraries:** Pandas, Plotly, Requests
* **Deployment:** Streamlit Community Cloud & GitHub

## 🚀 كيفية التشغيل (How to Run)
1. قم بعمل `Clone` للمستودع.
2. ثبت المكتبات اللازمة:
   ```bash
   pip install -r requirements.txt
