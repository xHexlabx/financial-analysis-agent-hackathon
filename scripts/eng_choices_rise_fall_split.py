import pandas as pd
import re # นำเข้า library สำหรับ Regular Expression

# --- 1. โหลดข้อมูลภาษาอังกฤษ ---
try:
    df_eng = pd.read_csv('eng_problems.csv')
    print(f"โหลดข้อมูลจาก 'eng_problems.csv' สำเร็จ พบ {len(df_eng)} รายการ")
except FileNotFoundError:
    print("ไม่พบไฟล์ 'eng_problems.csv' กรุณาตรวจสอบว่าไฟล์อยู่ในตำแหน่งที่ถูกต้อง")
    exit()

# --- 2. สร้างฟังก์ชันจำแนกประเภทสำหรับภาษาอังกฤษ ---
def classify_english_query(query: str) -> str:
    """
    จำแนกประเภทของ query ภาษาอังกฤษ โดยใช้ตรรกะตามลำดับความสำคัญ

    Args:
        query: ข้อความคำถามภาษาอังกฤษที่ต้องการจำแนก

    Returns:
        'rise-fall', 'choices', หรือ 'unclassified'
    """
    # ตรวจสอบให้แน่ใจว่า query เป็น string
    if not isinstance(query, str):
        return 'unclassified'

    # ================== รูปแบบ Regex สำหรับภาษาอังกฤษ ==================

    # 1. ตรวจสอบหาคีย์เวิร์ดที่ชัดเจนที่สุดของ "choices" ก่อน
    unambiguous_choices_pattern = r'Answer Choices:|Options:'
    if re.search(unambiguous_choices_pattern, query, re.IGNORECASE):
        return 'choices'

    # 2. หากไม่ใช่ ให้ตรวจสอบหาคีย์เวิร์ดของ "rise-fall"
    rise_fall_pattern = (
        r'rise or fall|increase or decrease|up or down|higher or lower'
    )
    if re.search(rise_fall_pattern, query, re.IGNORECASE):
        return 'rise-fall'

    # 3. หากยังไม่พบ ให้ใช้ "จำนวนตัวเลข" เป็นเกณฑ์ตัดสิน
    number_count = len(re.findall(r'\d+', query))
    NUMBER_THRESHOLD = 30
    if number_count > NUMBER_THRESHOLD:
        return 'rise-fall'

    # 4. เป็นขั้นตอนสุดท้าย ตรวจสอบหารูปแบบตัวเลือกทั่วไป
    # เพิ่ม ) เพื่อรองรับรูปแบบ "A)" หรือ "1)"
    general_choices_pattern = r'Which of the following|Select the best answer|[A-Z]\s*[:.)]|\d+\s*[:.)]'
    if re.search(general_choices_pattern, query):
        return 'choices'

    # ======================================================================

    # หากไม่ตรงกับเงื่อนไขใดเลย
    return 'unclassified'


# --- 3. เริ่มกระบวนการจำแนกประเภท ---
print("\n--- เริ่มกระบวนการจำแนกประเภทสำหรับภาษาอังกฤษ ---")

# ใช้ .apply() กับฟังก์ชันที่สร้างขึ้นสำหรับภาษาอังกฤษ
df_eng['classification'] = df_eng['query'].apply(classify_english_query)

print("--- กระบวนการจำแนกเสร็จสิ้น ---\n")


# --- 4. จัดการผลลัพธ์และบันทึกไฟล์ ---

# แสดงตัวอย่างผลลัพธ์การจำแนก
print("ตัวอย่างผลลัพธ์การจำแนก (ภาษาอังกฤษ):")
print(df_eng[['id', 'classification']].head())
print("-" * 30)

# ฟังก์ชันสำหรับกรองและบันทึกข้อมูล (ใช้ซ้ำได้)
def save_filtered_dataframe(dataframe, category, filename):
    filtered_df = dataframe[dataframe["classification"] == category]
    if not filtered_df.empty:
        df_to_save = filtered_df[['id', 'query']]
        df_to_save.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"บันทึกคำถาม {len(df_to_save)} ข้อ ({category}) ลงใน '{filename}' เรียบร้อยแล้ว")
    else:
        print(f"ไม่พบคำถามประเภท {category}")

# บันทึกไฟล์ที่คัดแยกแล้วสำหรับภาษาอังกฤษ
save_filtered_dataframe(df_eng, 'choices', 'eng_choices_problems.csv')
save_filtered_dataframe(df_eng, 'rise-fall', 'eng_rise_fall_problems.csv')
save_filtered_dataframe(df_eng, 'unclassified', 'eng_unclassified_problems.csv')
