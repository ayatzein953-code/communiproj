import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. تحميل البيانات
df = pd.read_csv("lte_data.csv")

# 2. إنشاء عمود الهدف (Target) بناءً على المؤشرات المتحركة والمستقبلية
# تحديد شرط الخطر الحالي بناءً على الوسيط المتحرك والانحراف المعياري والتشويش
is_bad_signal = (
    (df['RSRP_median_5'] < -112) | 
    (df['RSRP_roll_sd'] > 4.5) | 
    (df['RSSI_RSRP_diff'] > 15) | # يمكنك تعديل القيمة بناءً على بياناتك
    (df['RSRP_dist_gradient'] < -2.0)
).astype(int)

# نقل الهدف للمستقبل (التنبؤ بما سيحدث خلال الـ 3 خطوات القادمة)
df['breakdown_target'] = is_bad_signal.shift(-3).rolling(window=3).max().fillna(0)

# 3. تحديد الميزات (Features) التي تدرب النموذج
features = [
    'RSRP_num', 'RSRQ_num', 'SNR_num',  
    'RSRP_ma_5', 
    'RSRP_median_5', 
    'RSRP_roll_sd', 
    'RSRP_residual', 
    'RSRP_dist_gradient', 
    'RSSI_RSRP_diff'
]

# تنظيف البيانات من أي قيم فارغة ناتجة عن الـ Shift أو الـ Rolling
df = df.dropna(subset=features + ['breakdown_target'])

X = df[features].values
y = df['breakdown_target'].values



import numpy as np

# 1. التأكد أن كل الميزات رقمية ومعالجة النصوص
df[features] = df[features].apply(pd.to_numeric, errors='coerce')

# 2. استبدال أي قيمة "ما لانهاية" (inf أو -inf) بقيم فارغة (NaN)
df[features] = df[features].replace([np.inf, -np.inf], np.nan)

# 3. معالجة القيم الكبيرة جداً أو الـ NaN بملئها بالرقم 0 (أو يمكنك استخدام median)
df[features] = df[features].fillna(0)

# (اختياري إضافي): قص الأرقام الضخمة جداً إذا وجدت لتتأكد من عدم تجاوز الحدود
# df[features] = np.clip(df[features], -1e10, 1e10)

# 4. الآن قم بتجهيز X و الـ Scaling بأمان تام
X = df[features].values

print("تم تنظيف البيانات من الـ Infinity وتحجيمها بنجاح!")
# 4. توحيد مقياس البيانات (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




##########################################################


# # 1. التحقق من الأعمدة التي تحتوي على بيانات نصية غير رقمية
# for col in features:
#     # محاولة تحويل العمود إلى أرقام، وأي قيمة تعجز عن التحويل تتحول إلى NaN
#     non_numeric = pd.to_numeric(df[col], errors='coerce').isna()
#     if non_numeric.sum() > 0:
#         print(f"تنبيه: العمود '{col}' يحتوي على {non_numeric.sum()} قيمة غير رقمية (مثل '-' أو نصوص)!")
# 
# # 2. الحل الجذري: إجبار جميع أعمدة الـ features على أن تكون رقمية، واستبدال الشرطة أو النصوص بـ NaN
# for col in features:
#     df[col] = pd.to_numeric(df[col], errors='coerce')
# 
# # 3. ملء القيم الفارغة الناتجة (NaN) بـ 0 أو بالمتوسط (Mean) لكي لا تفسد النموذج
# df[features] = df[features].fillna(0)
# 
# # الآن أعد تعريف X وقم بعمل التوحيد بل أية مشاكل
# X = df[features].values
# X_scaled = scaler.fit_transform(X)
# print("تم تحجيم البيانات بنجاح!")
########################################################

# 5. تحويل البيانات إلى سلاسل زمنية (Time Steps) لتناسب LSTM
# def create_sequences(X_data, y_data, time_steps=10):
#     Xs, ys = [], []
#     for i in range(len(X_data) - time_steps):
#         Xs.append(X_data[i:(i + time_steps)])
#         ys.append(y_data[i + time_steps])
#     return np.array(Xs), np.array(ys)
import numpy as np

def create_sequences_fast(X_data, y_data, time_steps=10):
    # التأكد من أن البيانات مصفوفات NumPy لتحقيق أقصى سرعة
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)
    
    n_samples = len(X_data) - time_steps
    
    # استخدام تقطيع المصفوفات وزوايا الذاكرة (Strides) لتوليد المتسلسلات فوراً
    row_indices = np.arange(time_steps)[None, :] + np.arange(n_samples)[:, None]
    
    Xs = X_data[row_indices]
    ys = y_data[time_steps:]
    
    return Xs, ys




 TIME_STEPS = 10
 X_seq, y_seq = create_sequences_fast(X_scaled, y, time_steps=TIME_STEPS)

# 6. تقسيم البيانات إلى تدريب واختبار (بدون خلط عشوائي للحفاظ على التسلسل الزمني)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)

# 7. بناء نموذج LSTM
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid') # إخراج احتمال الانقطاع (بين 0 و 1)
])

# 8. تجميع النموذج (Compile)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# 9. تدريب النموذج (Training)
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X_test, y_test))S

# 10. تقييم أداء النموذجSS
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f"\n--- نتائج تقييم نموذج LSTM للتنبؤ بالانقطاع ---")
print(f"دقة النموذج (Accuracy): {accuracy * 100:.2f}%")
print(f"دقة التنبؤ (Precision): {precision * 100:.2f}%")
print(f"الاستدعاء (Recall): {recall * 100:.2f}%")


import joblib

# حفظ النموذج بعد التدريب
joblib.dump(model, 'my_model_lte.pkl')





import pandas as pd
non_numeric = df[pd.to_numeric(df['RSRQ'], errors='coerce').isna()]
print(non_numeric['RSRQ'].unique())
df["RSRQ_num"] = (
    df["RSRQ"]
    .astype(str)
    .str.strip()
    .replace(["-", ""], np.nan)
    .pipe(pd.to_numeric, errors="coerce")
)
##################3 لم يعمل الكود السابق 
print(type(["RSRQ"]))
print(df['RSRQ'].dtype)
print(df['RSRQ'].apply(type).unique())
print(df['RSRQ'].unique())
import numpy as np
import pandas as pd

# 1. تحويل العمود بالكامل إلى نص لتجنب أخطاء تداخل الأنواع (Strings & Numbers)
df["RSRQ"] = df["RSRQ"].astype(str).str.strip()

# 2. استبدال الشرطة المنفردة أو أي فراغات بقيم مفقودة NaN
df["RSRQ"] = df["RSRQ"].replace(["-", "", "nan", "NaN", "None"], np.nan)

# 3. تحويل العمود نهائياً إلى أرقام عشرية
df["RSRQ"] = pd.to_numeric(df["RSRQ"], errors="coerce")

# 4. التأكد من أن النوع أصبح رقمياً
print(df["RSRQ"].dtype)
print(df["RSRQ"].mean())
print(df["RSRQ_num"].dtype)
print((df["SNR_num"].dtype))
