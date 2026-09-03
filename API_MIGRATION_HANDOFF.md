# Handoff: หา route ที่หายไปใน API ตัวใหม่ (api.vrlive.io)

เอกสารนี้เขียนให้ session ที่จะไปเปิด **source code ของ API** อ่าน — ไม่ต้องมี context จากงานฝั่ง TTS
เป้าหมาย: หาว่าโครงสร้าง route ของ API ตัวใหม่ต่างจากตัวเดิมยังไง แล้วสรุปว่า client ฝั่ง TTS
ต้องยิงไปที่ path ไหน / ด้วย payload แบบไหน

---

## 1. สรุปปัญหาที่พบ (ยืนยันด้วยการยิงจริงแล้ว 2026-09-03)

บริการ TTS สองตัว (`:8010` voice-cloning และ `:8013` voxcpm+vc) เรียก API 2 endpoint:

| งาน | path ที่ client ใช้อยู่ |
|---|---|
| อัปโหลดไฟล์เสียงที่ render เสร็จ | `POST /api/v1/live-gpt/upload` |
| แจ้งผลกลับ n8n | `POST /api/v1/live-gpt/n8n/audio-callback` |

ผลการยิงจริง:

| host | path | ผล |
|---|---|---|
| `looklike.ai` | `POST /api/v1/live-gpt/upload` | **200** + `file_url` (ใช้งานได้) |
| `looklike.ai` | `GET /api/v1/live-gpt/n8n/audio-callback` | 404 `{"detail":"Not found"}` |
| `api.vrlive.io` | `POST /api/v1/live-gpt/upload` | **404** `{"success":false,"error":{"code":"NOT_FOUND","message":"Route not found",...}}` |
| `api.vrlive.io` | `GET /api/v1/live-gpt/n8n/audio-callback` | **404** (envelope เดียวกัน) |
| `api.vrlive.io` | `GET /api/v1/health` | **200** `{"status":"ok","uptimeMs":...}` |
| `api.vrlive.io` | `GET /` | nginx default page |
| `test.looklike.ai` | (ทุก path) | **526** — Cloudflare invalid SSL cert ที่ origin, ใช้ไม่ได้ |

**ข้อสังเกตสำคัญ: สอง host นี้เป็นคนละ stack ไม่ใช่โดเมนใหม่ของ service เดิม**

- `looklike.ai` → 404 shape เป็น `{"detail":"Not found"}` = **FastAPI / Python**
- `api.vrlive.io` → 404 shape เป็น `{"success":false,"error":{"code","message","retryable","requestId"}}` = **Node/Express-style** พร้อม requestId และ field `retryable`

เพราะฉะนั้นสมมติฐานที่ว่า "ย้ายโดเมน" น่าจะผิด — มันคือ **API คนละตัวที่เขียนใหม่** และ route ชุด
`live-gpt` อาจถูกจัดกลุ่ม/เปลี่ยนชื่อ/ยังไม่ได้ port มา

---

## 2. สิ่งที่ต้องหาใน source code

### 2.1 ยืนยันว่า repo ที่เปิดอยู่คือตัวไหน
ดูจาก error handler กลาง — ถ้ามี middleware ที่ตอบ
`{success:false, error:{code, message, retryable, requestId}}` แสดงว่านี่คือ **api.vrlive.io (ตัวใหม่)**
ถ้าเป็น FastAPI ที่ตอบ `{"detail":...}` แสดงว่าเป็น **looklike.ai (ตัวเดิม)**
ควรหาให้ครบทั้งสอง repo ถ้ามี เพื่อเทียบกัน

### 2.2 route table ปัจจุบัน
- ไล่หา router registration ทั้งหมด (เช่น `app.use(...)`, `router.post(...)`, `@app.post(...)`)
- **มี prefix `live-gpt` อยู่ไหม** — ถ้าไม่มี ชุด route นี้ถูกย้ายไปอยู่ใต้ prefix อะไรแทน
- version prefix ยังเป็น `/api/v1` หรือขยับเป็น `/api/v2` / ไม่มี version แล้ว
- ค้นคำเหล่านี้ในโค้ดทั้ง repo: `upload`, `live-gpt`, `livegpt`, `audio-callback`, `n8n`, `tts`, `voice`

### 2.3 endpoint อัปโหลด — ต้องได้ 4 ข้อนี้
client ปัจจุบันยิงแบบนี้ (ดู `voxcpm+vc/app/webhook.py:_upload` และ `voice-cloning/src/pipeline.py`):

```
POST <upload_url>
Authorization: Bearer <SIANGTTS_UPLOAD_TOKEN>
Content-Type: multipart/form-data
  field name = "file"   (filename = "<queue_id>.wav", mime = "audio/wav")
```
คาดหวัง response JSON ที่มี key **`file_url`** หรือ **`url`** (client อ่านสองตัวนี้เท่านั้น)

ต้องหาให้ได้ว่า:
1. **path ที่ถูกต้อง** ของ endpoint อัปโหลดใน API ตัวใหม่
2. **ชื่อ field ของ multipart** ยังเป็น `file` ไหม (บาง refactor เปลี่ยนเป็น `audio` / `files[]`)
3. **auth** — ยังเป็น `Authorization: Bearer <token>` ไหม หรือเปลี่ยนเป็น header อื่น / API key / ต้อง sign
   (token ที่ใช้อยู่ขึ้นต้น `n8n_...` ดูเหมือนออกมาสำหรับ n8n โดยเฉพาะ — เช็คว่าตัวใหม่ยัง issue token รูปแบบนี้ไหม)
4. **response schema** — key ชื่อ `file_url` เหมือนเดิมไหม ถ้าเปลี่ยน (เช่นเป็น `data.url`) client ต้องแก้ด้วย
   เพราะตอนนี้ถ้าไม่เจอ key จะ raise `"upload response has no file_url"`

เช็คเพิ่ม: **จำกัดชนิดไฟล์/ขนาดไหม** — ตัวเดิมรับ `audio/x-wav` ได้ (ยืนยันแล้ว) และเก็บลง
DigitalOcean Spaces (`sathu.sgp1.digitaloceanspaces.com/uploads/<uuid>.wav`)
ตัวใหม่ยังใช้ bucket เดิมหรือย้ายที่เก็บ

### 2.4 endpoint callback — เป็นตัวที่ยังคลุมเครือที่สุด
client ยิง:
```
POST <callback_url>
Content-Type: application/json
{"job_id": "...", "queue_id": "...", "file_url": "<url> หรือ \"none\"", "error": null หรือ "<ข้อความ>"}
```
ไม่มี auth header ใด ๆ

ต้องหาให้ได้ว่า:
1. endpoint ที่รับผลกลับจาก TTS **อยู่ที่ path ไหน** ใน API ตัวใหม่ — และมันเป็น route ของ API เอง
   หรือเป็นแค่ proxy ไป n8n webhook (ชื่อ `n8n/audio-callback` บอกใบ้ว่าอาจ proxy)
2. **body schema ที่มันคาดหวัง** ตรงกับ 4 field ข้างบนไหม โดยเฉพาะ:
   - `file_url` ตอน fail ส่งเป็น **string `"none"`** ไม่ใช่ `null` — ฝั่งรับ validate ตกไหม
   - มี field ที่ตัวใหม่บังคับเพิ่ม (เช่น `status`, `duration`, `signature`) หรือเปล่า
3. **ต้องมี auth ไหม** — ถ้าตัวใหม่บังคับ token แต่ client ไม่ได้ส่ง จะกลายเป็น 401 เงียบ ๆ
4. ยืนยันว่า **GET แล้วได้ 404 เพราะ method ไม่ตรง หรือเพราะไม่มี route จริง ๆ**
   (ที่ผ่านมาเช็คได้แค่ด้วย GET เพราะไม่อยาก trigger flow ปลายทาง — ใน source code ดูได้ตรง ๆ)

### 2.5 timeline ฝั่ง client (สืบจาก git แล้ว — ใช้เทียบกับ git log ฝั่ง API)

`git log -S "api.vrlive.io"` ในrepo TTS ให้ผลลัพธ์เดียว:

```
9fb2046b  2026-09-02 19:33  "add lazy mode"
```

โดเมน `api.vrlive.io` เข้ามาในโค้ดที่ track ครั้งแรกที่ commit นี้ ก่อนหน้านั้นทุก reference เป็น `looklike.ai`
สิ่งที่ commit นั้นเปลี่ยนเรื่องโดเมนมีแค่ 2 บรรทัด และอยู่ใน `voice-cloning/` (`:8010`) ทั้งคู่:

```diff
- "SIANGTTS_UPLOAD_URL", "https://looklike.ai/api/v1/live-gpt/upload"
+ "SIANGTTS_UPLOAD_URL", "https://api.vrlive.io/api/v1/live-gpt/upload"
- "https://test.looklike.ai/api/v1/live-gpt/n8n/audio-callback",
+ "https://api.vrlive.io/api/v1/live-gpt/n8n/audio-callback",
```

**สิ่งที่ต้องเทียบฝั่ง API:** ณ วันที่ 2026-09-02 (หรือก่อนหน้านั้น) repo ของ API ตัวใหม่มี route
`live-gpt` แล้วหรือยัง — ถ้า `git log` ฝั่ง API ไม่มี commit ที่เพิ่ม route ชุดนี้เลย แปลว่าฝั่ง client
ย้ายโดเมนไปก่อนที่ API จะพร้อม (เคส ก ข้างล่าง)

### 2.6 หาสาเหตุว่าทำไมถึงหาย
เลือกได้ 3 ทาง ต้องตอบให้ชัดว่าเป็นทางไหน:
- **(ก) ยังไม่ได้ port มา** — route ชุด live-gpt ยังอยู่แต่ใน repo เดิมเท่านั้น → ตัวใหม่ต้อง implement เพิ่ม
- **(ข) port มาแล้วแต่เปลี่ยน path/prefix** → แก้แค่ config ฝั่ง TTS
- **(ค) port มาแล้วแต่ยังไม่ deploy / ไม่ได้ mount router ใน entrypoint** → เช็ค entrypoint ว่า
  `require`/`import` router ครบไหม และดู deploy config / nginx upstream

ดู git log ของไฟล์ที่เกี่ยวข้องด้วย — อาจมี commit ที่ย้ายหรือลบ route ชุดนี้ พร้อมเหตุผลใน message

---

## 3. Deliverable ที่อยากได้กลับมา

1. ตาราง mapping: `path เดิม` → `path ใหม่` (หรือ "ยังไม่มี") ของทั้ง upload และ callback
2. diff ของ contract: multipart field name, auth header, response key, callback body schema
3. คำตอบข้อ 2.5 ว่าเป็นกรณี ก/ข/ค
4. ถ้าเป็นกรณี (ก) — ประเมินว่าต้องเพิ่มอะไรบ้างใน API ตัวใหม่ถึงจะรับของจาก TTS ได้

---

## 4. Context ฝั่ง TTS (ไว้เทียบ ไม่ต้องแก้อะไรใน repo API)

ค่า config ที่ client ใช้ อยู่ใน repo `SiangTTS+emotions/VoxCPM2+VC`:

| ที่ | ค่า |
|---|---|
| `voxcpm+vc/.env:54` | `SIANGTTS_UPLOAD_URL` — ตอนนี้ตั้งกลับไปที่ `looklike.ai` แล้วเพื่อให้ใช้งานได้ |
| `voxcpm+vc/.env:58` | `SIANGTTS_DEFAULT_CALLBACK` — **ยังชี้ `api.vrlive.io` = ยังพัง** |
| `voxcpm+vc/app/config.py:60,63` | default ในโค้ด ยังเป็น `looklike.ai` / `test.looklike.ai` |
| `voice-cloning/.env:20` | `SIANGTTS_UPLOAD_URL` = `looklike.ai` (ไม่เคยถูกเปลี่ยน จึงไม่เคยพัง) |
| `voice-cloning/src/webhook.py:76` | default callback = `api.vrlive.io` |

**กับดักที่ต้องรู้:** `_post_callback` ทั้งสองบริการ **ไม่เช็ค status code ของ response เลย**
(`await client.post(url, json=payload)` แล้วจบ ไม่มี `raise_for_status()`)
แปลว่า callback ที่ 404/401 จะเงียบสนิท และงานยังขึ้นสถานะ `completed` ตามปกติ —
ถ้าเจอว่า callback path ผิดจริง ให้แจ้งกลับด้วย เพราะฝั่ง TTS ต้องเพิ่มการเช็ค status ควบคู่กับการแก้ URL
