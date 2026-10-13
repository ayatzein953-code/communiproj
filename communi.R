library(tidyverse)
library(fs)

main_dir <- "C:/Users/ayaat/Desktop/communiproj/Dataset"

csv_files <- dir_ls(main_dir, recurse = TRUE, glob = "*.csv")

fixed_data <- csv_files %>% 
  set_names() %>% 
  imap(~ {
    # قراءة كل الأعمدة كنصوص (col_character) لمنع أي تعارض نهائياً
    read_csv(.x, col_types = cols(.default = col_character()), show_col_types = FALSE) %>% 
      mutate(#source_file = basename(.x)
            source_folder = basename(path_dir(.x)))
  }) %>% 
  bind_rows()

# معاينة البيانات بعد الدمج
glimpse(fixed_data)







######################################التاكد من تسلسل زمني صحيح 
check_time_series <- function(df, time_col) {
  
  # تحويل النص إلى تاريخ ووقت بناءً على الشكل الموجود في ملفاتك (YYYY.MM.DD_HH.MM.SS)
  times <- as.POSIXct(df[[time_col]], format = "%Y.%m.%d_%H.%M.%S")
  
  # التحقق من أن التحويل تم بنجاح ولم يترك قيماً فارغة
  if (all(is.na(times))) {
    stop("فشل تحويل التاريخ! يرجى التأكد من تطابق الصيغة.")
  }
  
  # فحص الترتيب الزمني
  is_ordered <- !is.unsorted(times)
  
  if (!is_ordered) {
    warning("البيانات غير مرتبطة زمنياً بشكل صحيح! جاري إعادة ترتيبها تصاعدياً...")
    df <- df[order(times), ]
  } else {
    message("التسلسل الزمني مرتب بشكل ممتاز.")
  }
  
  return(df)
}

# الآن سيتم تشغيل الدالة بسلاسة دون أخطاء:
fixed_data <- check_time_series(fixed_data, "Timestamp")
###################################################33



library(imputeTS)
lte_data <- fixed_data %>%
    # تطبيق مرشح كالمان على قوة الإشارة وجودتها
   mutate( NRxRSRP = suppressWarnings(as.numeric(NRxRSRP)),
           NRxRSRP = na_kalman(NRxRSRP),
           
           NRxRSRQ = suppressWarnings(as.numeric(NRxRSRQ)),
           NRxRSRQ = na_kalman(NRxRSRQ),
           
           
           # تطبيق الاستيفاء الخطي على RSSI لأن النقص قليل فيه
           RSSI = suppressWarnings(as.numeric(RSSI)),
           RSSI = na_interpolation(RSSI, option = "linear")
           
           
           
            )
# فحص القيم غير الرقمية في عمود NRxRSRP مثلاً
# fixed_data %>% 
#   mutate(temp_num = suppressWarnings(as.numeric(NRxRSRP))) %>% 
#   filter(is.na(temp_num) & !is.na(NRxRSRP)) %>% 
#   distinct(NRxRSRP)


lte_data <- lte_data %>%
         drop_na(ServingCell_Lon, ServingCell_Lat, ServingCell_Distance )

# 4. التحقق من عدم وجود أي قيم ناقصة (NA) متبقية
 missing_count<- sum(is.na(lte_data))
 message(paste("اجمالي القيم الباقية الناقصة " ,missing_count ))
 
 
 ################3العمليات الاحصائية   
 
 


 
 
 # 2. اكتشاف القيم الشاذة (Outlier Detection) باستخدام طريقة IQR
 detect_outliers <- function(x) {
   Q1 <- quantile(x, 0.25, na.rm = TRUE)
   Q3 <- quantile(x, 0.75, na.rm = TRUE)
   IQR_val <- Q3 - Q1
   lower_bound <- Q1 - 1.5 * IQR_val
   upper_bound <- Q3 + 1.5 * IQR_val
   return(x < lower_bound | x > upper_bound)
 }
 

#####33تحويل ال 200 الى na 
 #lte_data$RSRP_num[lte_data$RSRP == -200] <- NA
  
 
 # معرفة عدد القيم المفقودة في عمود RSRP
 sum(is.na(lte_data$RSRP_num))
 
 # معرفة نسبتها المئوية من إجمالي البيانات
 mean(is.na(lte_data$RSRP_num)) * 100

 library(dplyr)
 
 lte_data <- lte_data %>%
   mutate(
     # 1. إزالة أي مسافات مخفية (trimws) ثم التحويل لأرقام بشكل آمن
     RSRP_num = as.numeric(trimws(RSRP)),
     RSRQ_clean = trimws(RSRQ),
     RSRQ_clean = na_if(RSRQ_clean , "-"),
     Timestamp = trimws(Timestamp),
     Timestamp_clean = na_if(Timestamp , "-"),
     RSSI_clean = trimws(RSSI),
     RSSI_clean = na_if(RSSI_clean , "-"),
     RSRQ_num =suppressWarnings(as.numeric(RSRQ_clean)),
     RSSI_num =suppressWarnings(as.numeric(RSSI_clean)),
     Timestamp_num =suppressWarnings(as.numeric(Timestamp_clean)),
    
     
     RSSI_num =ifelse(RSSI_num > -30 | RSSI_num < -120, NA, RSSI_num),
     RSSI_num =na_kalman(RSSI_num),
     RSSI_num = na_locf(RSSI_num, option ="both"),
   
     
     # 2. تحويل القيم -200 فقط إلى NA، وترك باقي الأرقام كما هي
     RSRP_num = ifelse(RSRP_num == -200, NA, RSRP_num)
   
   )
 
 
# 1. حساب الإحصائيات الوصفية والوسط الحسابي والمدى الطبيعي
 library(dplyr)
 library(ggplot2)
 kpi_summary <- lte_data %>%
   summarise(
     across(c(RSRP_num , RSRQ_num, SNR_num ), 
            list(
              Mean = ~mean(., na.rm = TRUE),
              Median = ~median(., na.rm = TRUE),
              Min = ~min(., na.rm = TRUE),
              Max = ~max(., na.rm = TRUE),
              SD = ~sd(., na.rm = TRUE)
            ))
   )
 
 print(kpi_summary)
 
 
 # إضافة عمود يوضح ما إذا كان السطر يحتوي على قيمة شاذة في الـ RSRP مثلاً
 lte_data <- lte_data %>%
   mutate(is_outlier_RSRP = detect_outliers(as.numeric(RSRP_num)),
          is_outlier_RSRQ = detect_outliers(as.numeric(RSRQ_num)),
          is_outlier_RSSI = detect_outliers(as.numeric( RSSI_num )))
 
 # عدد القيم الشاذة المكتشفة
 table(lte_data$is_outlier_RSRP)
 table(lte_data$is_outlier_RSRQ)
 table(lte_data$is_outlier_RSSI)
 
 outliers_sample <- lte_data[lte_data$is_outlier_RSRP == TRUE, ]
 head(outliers_sample)
 
 outliers_sample <- lte_data[lte_data$is_outlier_RSSI == TRUE, ]
 head(outliers_sample)
 ###########لنعرف ازا  في قيم ناتجة عن خطأ بالقياس مثل 200 
 # معرفة القيم الفريدة الموجودة داخل القيم الشاذة
 unique(lte_data$RSRP[lte_data$is_outlier_RSRP == TRUE])
 
 # 3. التحقق من عدد القيم المفقودة الفعلي الآن
 # sum(is.na(lte_data$RSRP_num))
 # 
 # RSRP_num = na_kalman(lte_data$RSRP_num)
 # 
 # sum(is.na(lte_data$RSRP_num))

 # لمعرفة مواقع الـ NA في العمود
 which(is.na(lte_data$RSRP_num))
 
 library(imputeTS)
 
 lte_data <- lte_data %>%
   mutate(
     # تطبيق مرشح كالمان أولاً
     RSRP_num = na_kalman(RSRP_num),
     RSRQ_num = na_kalman(RSRQ_num),
     # ملء أي فراغات متبقية (مثل البداية والنهاية أو الفجوات الطويلة) 
     # بتكرار أقرب قيمة صحيحة سابقة أو لاحقة
     RSRP_num = na_locf(RSRP_num, option = "both"),
     RSRQ_num = na_locf(RSRQ_num, option = "both"),
   )
 sum(is.na(lte_data$RSRP_num))
 sum(is.na(lte_data$RSRQ_num))
 
 

 # summary(lte_data$RSRQ)
 # unique(lte_data$RSRQ)
 # إضافة عمود يوضح ما إذا كان السطر يحتوي على قيمة شاذة في الـ RSRP مثلاً
 summary(lte_data$RSSI)
 unique(lte_data$RSSI)
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 library(dplyr)
 library(zoo)
 
 lte_data <- lte_data %>%
   # 1. ترتيب البيانات تسلسلياً (حسب الوقت أو التسلسل) لضمان دقة الحسابات المتحركة
   arrange(Timestamp) %>% 
   
   mutate(
     # 2. المتوسط المتحرك والوسيط المتحرك (Rolling Mean & Median) لتخفيف الضوضاء
     RSRP_ma_5 = rollapply(RSRP_num, width = 5, FUN = mean, fill = NA, align = "right"),
    RSRP_median_5 = rollapply(RSRP_num, width = 5, FUN = median, fill = NA, align = "right"),
     
     # 3. الانحراف المعياري المتحرك (Rolling Standard Deviation) لقياس التذبذب والاستقرار
     RSRP_roll_sd = rollapply(RSRP_num, width = 5, FUN = sd, fill = NA, align = "right"),
     
     # 4. الفارق بين القيمة الحالية والمتوسط المتحرك (Residuals / الفارق اللحظي عن النمط العام)
     RSRP_residual = RSRP_num - RSRP_ma_5,
     
     # 5. معدل التغير بالنسبة للمسافة (Distance Gradient) - تأكد أن اسم عمود المسافة لديك هو distance
    distance_clean = trimws(ServingCell_Distance),
    distance_clean = na_if(distance_clean , "-"),
    distance_num =suppressWarnings(as.numeric(distance_clean)),
    
     RSRP_dist_gradient = (RSRP_num - lag(RSRP_num)) / (distance_num - lag(distance_num)),
     
     # 6. التداخل أو الفارق بين RSSI و RSRP كمؤشر إضافي للـ KPIs
    RSSI_RSRP_diff = RSSI_num - RSRP_num
   )
  summary(lte_data)
  
  
  
  colSums(is.na(lte_data))
  #####3حدف الاعمدة المؤقتة 
  lte_data <- lte_data %>%
  select(-ends_with("_clean"), -starts_with("is_outlier_"))
  summary(lte_data)
  
  
  write.csv(lte_data  ,"lte_data.csv" , row.names = FALSE)
  
  library(tidyverse)
  library(ggplot2)
  library(dplyr)
  setwd("C:/Users/ayaat/Desktop/cyclingpro")
  lte_data <- read_csv("lte_data.csv")%>%
    mutate(SNR = trimws(SNR),
           SNR_clean = na_if(SNR , "-"),
           SNR_num =suppressWarnings(as.numeric(SNR_clean)),)
  typeof(lte_data$SNR_num)
  write.csv(lte_data , "lte_data.csv" , row.names = FALSE)

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 



