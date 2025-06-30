import pandas as pd
import re

# โหลดไฟล์ CSV เข้าสู่ DataFrame
df = pd.read_csv('./data/test.csv')

# ฟังก์ชันสำหรับตรวจจับภาษาโดยการตรวจสอบอักขระไทย (เหมือนเดิม)
def detect_language_thai_english(text):
    # ตรวจสอบว่ามีอักขระไทยอยู่ในช่วง Unicode (0x0E00-0x0E7F) หรือไม่
    if any(0x0E00 <= ord(char) <= 0x0E7F for char in text):
        return 'Thai'
    else:
        return 'English'

# ใช้ฟังก์ชันเพื่อสร้างคอลัมน์ใหม่ 'language' ใน DataFrame
df['language'] = df['query'].apply(detect_language_thai_english)

# แยก DataFrame ออกเป็นสองส่วนตามภาษา
thai_problems_df = df[df['language'] == 'Thai'].drop(columns=['language'])
eng_problems_df = df[df['language'] == 'English'].drop(columns=['language'])

# บันทึกแต่ละส่วนเป็นไฟล์ CSV ใหม่
thai_problems_df.to_csv('./data/thai_problems.csv', index=False)
eng_problems_df.to_csv('./data/eng_problems.csv', index=False)

print("ทำการแยกไฟล์และบันทึกเรียบร้อยแล้ว:")
print(f"- 'thai_Problems.csv' จำนวน {len(thai_problems_df)} แถว")
print(f"- 'eng_Problems.csv' จำนวน {len(eng_problems_df)} แถว")

print("\nคุณสามารถตรวจสอบไฟล์เหล่านี้ได้ในส่วน 'Files' ของสภาพแวดล้อมที่คุณใช้งาน (เช่น Google Colab)")
