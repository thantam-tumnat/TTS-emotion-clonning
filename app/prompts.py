"""
Prompts and few-shot examples for Thai TTS Tone Annotation.
"""

SYSTEM_PROMPT = """คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์อารมณ์และน้ำเสียงของข้อความภาษาไทยเพื่อใช้ในการสังเคราะห์เสียงอ่าน (TTS Tone Annotation)

ภารกิจของคุณ:
รับรายการข้อความย่อย (clauses) ที่ตัดไว้พร้อม index และระบุน้ำเสียง (tone) และระดับความเข้มข้น (intensity) สำหรับแต่ละ index ผ่านเครื่องมือ annotate_clauses

กฎเหล็กในการวิเคราะห์:
1. ต้องระบุ label ให้ครบทุก index ที่ได้รับอย่างแม่นยำ ไม่ขาดและไม่เกิน
2. เลือก tone จาก enum ทั้ง 8 ค่านี้เท่านั้น:
   - neutral: น้ำเสียงปกติ เป็นกลาง บรรยาย ข้อมูลทั่วไป
   - sad: เศร้า เสียใจ ผิดหวัง สะเทือนใจ ขอโทษจากใจจริง
   - happy: ดีใจ ร่าเริง มีความสุข ยิ้มแย้ม ยินดี
   - angry: โกรธ ไม่พอใจ เสียงแข็ง ดุดัน ตำหนิ
   - excited: ตื่นเต้น กระตือรือร้น ดีใจสุดขีด เร่งเร้า
   - calm: สงบ สบาย ผ่อนคลาย นุ่มนวล พูดช้า มีสติ
   - nervous: ประหม่า ลังเล ไม่มั่นใจ กลัว หวาดระแวง
   - sarcastic: ประชด ประชัน แดกดัน พูดอย่างแต่หมายถึงอีกอย่าง
3. หากไม่มั่นใจ หรือไม่มีอารมณ์ชัดเจน ให้เลือก neutral เสมอ (ปลอดภัยกว่าใส่อารมณ์ผิด)
4. ถ้าทั้งข้อความสื่อถึงอารมณ์เดียวกันตลอด ให้ใช้ tone เดียวกันทุก index ไม่จำเป็นต้องพยายามหาความหลากหลาย
5. ตัดสินจากความหมายและบริบทโดยรวม ไม่ตัดสินจากคำเดี่ยวๆ
6. intensity เป็นจำนวนเต็ม 1, 2 หรือ 3:
   - 1 = เล็กน้อย / แผ่วเบา (slightly)
   - 2 = ปกติ / ชัดเจน (moderate - ค่ามาตรฐาน)
   - 3 = รุนแรง / มาก (very / strongly)
   ใช้ 2 เป็นค่าปกติ ใช้ 1 หรือ 3 เฉพาะเมื่อบริบทระบุความชัดเจนอย่างมากเท่านั้น

คำเตือนด้านความปลอดภัย:
ห้ามส่งข้อความกลับมาในผลลัพธ์โดยเด็ดขาด ให้ส่งกลับมาเฉพาะ index (i), tone, และ intensity เท่านั้น"""

FEW_SHOT_EXAMPLES = [
    {
        "description": "เคส 1: โทนเดียวทั้งก้อน (sad ตลอด)",
        "input": {
            "clauses": [
                {"i": 0, "text": "ฉันคิดถึงเธอเหลือเกิน "},
                {"i": 1, "text": "ทำไมเรื่องมันต้องจบลงแบบนี้ด้วย"}
            ]
        },
        "output": {
            "labels": [
                {"i": 0, "tone": "sad", "intensity": 2},
                {"i": 1, "tone": "sad", "intensity": 2}
            ]
        }
    },
    {
        "description": "เคส 2: เปลี่ยนโทนกลางข้อความ (sad -> angry)",
        "input": {
            "clauses": [
                {"i": 0, "text": "ขอโทษนะ "},
                {"i": 1, "text": "ฉันไม่ได้ตั้งใจ "},
                {"i": 2, "text": "แต่เธอก็ไม่ฟังฉันเลย"}
            ]
        },
        "output": {
            "labels": [
                {"i": 0, "tone": "sad", "intensity": 2},
                {"i": 1, "tone": "sad", "intensity": 2},
                {"i": 2, "tone": "angry", "intensity": 2}
            ]
        }
    },
    {
        "description": "เคส 3: ข้อมูลข่าวสาร / คำอธิบาย เป็นกลางล้วน (neutral)",
        "input": {
            "clauses": [
                {"i": 0, "text": "กรมอุตุนิยมวิทยาประกาศเตือน "},
                {"i": 1, "text": "จะมีฝนตกหนักถึงหนักมากในหลายพื้นที่ "},
                {"i": 2, "text": "ประชาชนควรระมัดระวังน้ำท่วมฉับพลัน"}
            ]
        },
        "output": {
            "labels": [
                {"i": 0, "tone": "neutral", "intensity": 2},
                {"i": 1, "tone": "neutral", "intensity": 2},
                {"i": 2, "tone": "neutral", "intensity": 2}
            ]
        }
    },
    {
        "description": "เคส 4: ประชดประชัน (sarcastic)",
        "input": {
            "clauses": [
                {"i": 0, "text": "แหม เก่งจังเลยนะ "},
                {"i": 1, "text": "ทำพังหมดทั้งห้องแล้วเนี่ย"}
            ]
        },
        "output": {
            "labels": [
                {"i": 0, "tone": "sarcastic", "intensity": 2},
                {"i": 1, "tone": "sarcastic", "intensity": 2}
            ]
        }
    }
]

ANNOTATE_TOOL = {
    "name": "annotate_clauses",
    "description": "Annotate emotional tone and intensity for each clause index.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {
                            "type": "integer",
                            "description": "The exact clause index matching the input clause"
                        },
                        "tone": {
                            "type": "string",
                            "enum": [
                                "neutral",
                                "sad",
                                "happy",
                                "angry",
                                "excited",
                                "calm",
                                "nervous",
                                "sarcastic"
                            ],
                            "description": "The emotional tone"
                        },
                        "intensity": {
                            "type": "integer",
                            "enum": [1, 2, 3],
                            "description": "Intensity level (1=slightly, 2=standard, 3=very)"
                        }
                    },
                    "required": ["i", "tone", "intensity"],
                    "additionalProperties": False
                },
                "description": "List of clause label annotations"
            }
        },
        "required": ["labels"],
        "additionalProperties": False
    }
}
