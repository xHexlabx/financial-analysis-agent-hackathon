import pandas as pd
import re # นำเข้า library สำหรับ Regular Expression

# --- 1. โหลดข้อมูล ---
try:
    df = pd.read_csv('./data/thai_problems.csv')
    print(f"โหลดข้อมูลจาก 'thai_problems.csv' สำเร็จ พบ {len(df)} รายการ")
except FileNotFoundError:
    print("ไม่พบไฟล์ 'thai_problems.csv' กรุณาตรวจสอบว่าไฟล์อยู่ในตำแหน่งที่ถูกต้อง")
    exit()

# --- 2. สร้างฟังก์ชันจำแนกประเภทด้วยตรรกะที่ปรับปรุงลำดับความสำคัญใหม่ ---
def classify_with_refined_logic(query: str) -> str:
    """
    จำแนกประเภทของ query โดยใช้ตรรกะตามลำดับความสำคัญที่ปรับปรุงใหม่

    Args:
        query: ข้อความคำถามที่ต้องการจำแนก

    Returns:
        'rise-fall', 'choices', หรือ 'unclassified'
    """
    # ========================= ตรรกะตามลำดับความสำคัญใหม่ =========================

    # 1. ตรวจสอบหาคีย์เวิร์ดที่ชัดเจนที่สุดของ "choices" ก่อน
    # คำว่า "ตัวเลือกคำตอบ:" มีความสำคัญสูงสุด
    unambiguous_choices_pattern = r'ตัวเลือกคำตอบ:'
    if re.search(unambiguous_choices_pattern, query):
        return 'choices'

    # 2. หากไม่ใช่ ให้ตรวจสอบหาคีย์เวิร์ดของ "rise-fall"
    rise_fall_pattern = (
        r'ขึ้นหรือลง|สูงขึ้นหรือต่ำลง|ปรับตัวขึ้นหรือลง|เพิ่มขึ้นหรือลดลง|'
        r'rise or fall|increase or decrease|up or down'
    )
    if re.search(rise_fall_pattern, query, re.IGNORECASE):
        return 'rise-fall'

    # 3. หากยังไม่พบ ให้ใช้ "จำนวนตัวเลข" เป็นเกณฑ์ตัดสิน
    number_count = len(re.findall(r'\d+', query))
    NUMBER_THRESHOLD = 100
    if number_count > NUMBER_THRESHOLD:
        # ในขั้นนี้ เราสันนิษฐานว่าถ้ามีตัวเลขเยอะและไม่ใช่ rise-fall จากคีย์เวิร์ด
        # มันก็ไม่ควรเป็น choices เช่นกัน แต่เพื่อความปลอดภัย เราจะยังคงตรรกะนี้ไว้
        return 'rise-fall'

    # 4. เป็นขั้นตอนสุดท้าย ตรวจสอบหารูปแบบตัวเลือกทั่วไป
    general_choices_pattern = r'จงเลือกคำตอบ|ข้อใด|โปรดตอบด้วยตัวเลือก|[A-Zก-ฮ]\s*[:.]|\d+\s*[:.]'
    if re.search(general_choices_pattern, query):
        return 'choices'

    # ======================================================================

    # หากไม่ตรงกับเงื่อนไขใดเลย
    return 'unclassified'


# --- 3. เริ่มกระบวนการจำแนกประเภท ---
print("\n--- เริ่มกระบวนการจำแนกประเภทด้วยตรรกะที่ละเอียดขึ้น ---")

# ใช้ .apply() กับฟังก์ชันที่ปรับปรุงใหม่
df['classification'] = df['query'].apply(classify_with_refined_logic)

print("--- กระบวนการจำแนกเสร็จสิ้น ---\n")


# --- 4. จัดการผลลัพธ์และบันทึกไฟล์ ---

# แสดงตัวอย่างผลลัพธ์การจำแนก
print("ตัวอย่างผลลัพธ์การจำแนก:")
print(df[['id', 'classification']].head())
print("-" * 30)

# ลองทดสอบกับคำถามที่ยกมาโดยเฉพาะ
problematic_query = """ตอบคำถามด้วยตัวเลือกที่เหมาะสม A, B, C และ D กรุณาตอบด้วยคำตอบที่ถูกต้อง A, B, C หรือ D เท่านั้น อย่าใช้คำฟุ่มเฟือยหรือให้ข้อมูลเพิ่มเติม คำถาม: Skytop Co. ซึ่งเป็นหน่วยงานที่ไม่แสวงหาผลกำไรกำลังพิจารณาที่จะซื้อเครื่องจักรราคา $80000 ซึ่งจะสร้างกระแสเงินสดเข้าจำนวน $25000 เป็นเวลาสี่ปี Skytop ประเมินโครงการลงทุนโดยใช้กระแสเงินสดคิดลดด้วยต้นทุนเงินทุน 10% ต่อปี จากตารางต่อไปนี้ Skytop ควรดำเนินการอย่างไรเกี่ยวกับการซื้อเครื่องจักรและเพราะเหตุใด? มูลค่าอนาคตของ $1 เป็นเวลา 4 ปี ที่ 10% $1.464 มูลค่าปัจจุบันของ $1 เป็นเวลา 4 ปี ที่ 10% $0.683 มูลค่าอนาคตของเงินงวดสามัญ $1 เป็นเวลา 4 ปี ที่ 10% $4.641 มูลค่าปัจจุบันของเงินงวดสามัญ $1 เป็นเวลา 4 ปี ที่ 10% $3.170 ซื้อ: เหตุผล: ตัวเลือกคำตอบ: A: ใช่ กระแสเงินสดสุทธิคือ $20000, B: ใช่ มูลค่าอนาคตสุทธิคือ $36025, C: ไม่ มูลค่าปัจจุบันสุทธิคือ ($750), D: ไม่ มูลค่าปัจจุบันสุทธิคือ ($8750) คำตอบ:"""
test_result = classify_with_refined_logic(problematic_query)
print(f"ผลการทดสอบคำถามที่แยกผิด: '{test_result}' (ถูกต้องแล้ว!)")
print("-" * 30)


# ฟังก์ชันสำหรับกรองและบันทึกข้อมูล
def save_filtered_dataframe(dataframe, category, filename):
    filtered_df = dataframe[dataframe["classification"] == category]
    if not filtered_df.empty:
        df_to_save = filtered_df[['id', 'query']]
        df_to_save.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"บันทึกคำถาม {len(df_to_save)} ข้อ ({category}) ลงใน '{filename}' เรียบร้อยแล้ว")
    else:
        print(f"ไม่พบคำถามประเภท {category}")

# บันทึกไฟล์ที่คัดแยกแล้ว
save_filtered_dataframe(df, 'choices', './data/thai_choices_problems.csv')
save_filtered_dataframe(df, 'rise-fall', './data/thai_rise_fall_problems.csv')
save_filtered_dataframe(df, 'unclassified', './data/thai_unclassified_problems.csv')
