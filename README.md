# taz-mltrain — ML สำหรับวิเคราะห์การออกออเดอร์ SSS (XAUUSD M5)

ไปป์ไลน์ Python ที่จำลองลอจิกการเข้าออเดอร์ของ EA **`TAZ Expert Strategy-gemini.mq4` v7.22**
(repo `tazexpert/tazexpert-mp4`) แบบแท่งต่อแท่งบนไฟล์ `XAUUSD_M5.csv` แล้ว

1. สร้างชุดข้อมูลของ "จุดที่ EA จะได้สัญญาณ" ทุกจุด (SSS pullback entry + Reject+Rebound)
2. จำลองผลออเดอร์ตามที่ EA บริหารจริง (TP1/TP2, Auto SL 2 ขั้น, Double-Reject Close, News Protection)
3. Back-test ตามกฎ EA (session, re-entry, cooldown, spike filter, S/R block, Guard 3)
4. เทรนโมเดล (HistGradientBoosting) ทำนายว่า TP1 จะชนะหรือไม่ ด้วย walk-forward แบบ purge (ไม่มี look-ahead)
5. แสดงว่าการกรองสัญญาณด้วยความน่าจะเป็นจากโมเดลช่วยเพิ่ม expectancy (หน่วย R) ได้หรือไม่

## วิธีใช้

```bash
pip install -r requirements.txt
```

วางไฟล์ข้อมูลไว้ที่ `data/XAUUSD_M5.csv` (รองรับ MT4 History Center export แบบไม่มี header,
MT5 export แบบ tab `<DATE> <TIME> <OPEN> ... <SPREAD>` หรือ CSV ที่มีคอลัมน์ Date/Time) แล้วรัน:

```bash
python -m taz_ml.train --csv data/XAUUSD_M5.csv --out output
```

ตัวเลือกที่สำคัญ:

| option | ความหมาย |
|---|---|
| `--point 0.001` | MODE_POINT ของโบรกเกอร์ (ค่าเริ่มต้น: เดาจากจำนวนทศนิยมใน CSV) — **ต้องตรงกับโบรกที่ EA รันจริง** เพราะ input ของ EA เป็นหน่วย point |
| `--spread 0.20` | spread เป็น $ (ใช้เมื่อ CSV ไม่มีคอลัมน์ spread) |
| `--max-hold 864` | ปิดออเดอร์ที่ค้างเกิน N แท่ง M5 |

เวลาใน CSV ต้องเป็น **Server Time ของโบรก** เดียวกับที่ตั้ง session ใน EA

ผลลัพธ์ใน `output/`: `dataset.csv` (ทุกสัญญาณ + ผล + สถานะที่ EA ทำ), `features.csv`, `ea_trades.csv`
(ออเดอร์ที่ EA จะเปิดจริง), `oos_predictions.csv`, `model.joblib`, `report.json`

ให้คะแนนสัญญาณล่าสุดด้วยโมเดลที่เทรนแล้ว:

```bash
python -m taz_ml.score --csv data/XAUUSD_M5.csv --model output/model.joblib --last 20 --threshold 0.55
```

ทดสอบความถูกต้องของการพอร์ต (เทียบกับการพอร์ตแบบ line-by-line ของ MQ4 + ตรวจ look-ahead):

```bash
python -m pytest -q tests
```

ทดลองกับข้อมูลสังเคราะห์ (ไม่ใช่ข้อมูลตลาดจริง — ผลไม่มีความหมาย ใช้ทดสอบว่าโค้ดทำงาน):

```bash
python tools/make_synthetic.py data/XAUUSD_M5_synthetic.csv 365
```

```bash
python -m taz_ml.train --csv data/XAUUSD_M5_synthetic.csv --out output/synthetic
```

## การอ่านผล

- **Label** `y` = ขา TP1 ปิดกำไร (`tp1_r > 0`) — รวมกรณี Auto SL ล็อกกำไรแล้วโดนชน
- **`set_r`** = ผลทั้งชุดเป็นหน่วย R (R = ระยะ Entry→SL) ถ่วง 0.5·TP1 + 0.5·TP2 ตามที่ EA แบ่ง lot ครึ่งๆ
- ตาราง `oos_threshold_table`: ถ้าเลือกเฉพาะสัญญาณที่ `p_win ≥ threshold` จะได้ win rate / avg R เท่าไร
  ทั้งกับ "ทุกสัญญาณ" และ "เฉพาะที่ EA เปิดจริง" — ให้ดูว่าดีกว่า baseline แบบ **สม่ำเสมอทุก fold** หรือไม่
  ถ้า AUC ราว 0.5 แปลว่าโมเดลไม่มี edge อย่าเอาไปใช้
- การกรองออเดอร์ที่ EA เปิดจริงเป็นค่าประมาณ: ในของจริงถ้าข้ามออเดอร์หนึ่งไป session นั้นอาจได้ออเดอร์อื่นแทน

## สมมติฐาน / ข้อจำกัดของการจำลอง

- จำลองระดับแท่ง M5: ถ้า SL กับ TP อยู่ในแท่งเดียวกัน ถือว่า **โดน SL ก่อน** (conservative);
  การเลื่อน SL จาก Auto SL มีผลตั้งแต่แท่งถัดไป
- เข้าออเดอร์ที่ open ของแท่งถัดไป (BUY = Bid + spread) ราคาใน CSV ถือเป็น Bid
- ไม่ได้จำลอง: Guard 2 (Daily Max Loss เป็น USD), การทบ lot หลังโดน SL, `_fix_TP1`, การซ่อม TP2 ที่หาย,
  stop level ของโบรก, slippage, กลยุทธ์ M15 Swing / M15 Rejection (ปิดอยู่โดย default)
- กล่อง Sideway M15 / Guard H1 สร้างจากการ resample M5 (อาจต่างจากแท่ง M15/H1 ของโบรกเล็กน้อยช่วงข้อมูลขาด)

## ข้อสังเกตจากการตรวจโค้ด EA (`TAZ Expert Strategy-gemini.mq4` v7.22)

ยืนยันจากการอ่านโค้ดและจากตัวจำลอง:

1. **S/R block ถูก Reject+Rebound ข้ามในแท่งเดียวกัน** (`OnTick` main path → `CheckSSSRejectRebound`, ~บรรทัด 4170)
   เมื่อใช้ `InpSSSUsePullbackEntry=true` สัญญาณหลักเกิดที่แท่ง resumption อยู่แล้ว พอโดน `SSS_IsBlockedBySRZone()`
   บล็อก ใน tick เดียวกัน RR จะสแกน breakout เดิม เจอ resumption ที่ `i==1` แท่งเดียวกัน และเปิดออเดอร์ทันที
   (ถ้า Close ของแท่ง breakout ก็อยู่เหนือ/ใต้ขอบกล่องด้วย) — ฟิลเตอร์ S/R และ cooldown 6 แท่งแทบไม่มีผล
   ในข้อมูลทดสอบ 44 จาก 71 ครั้งที่โดนบล็อกถูก RR เปิดแทนในแท่งเดียวกัน
2. **SL ขั้นต่ำ $7 ไม่ได้ถูกบังคับจริง** (`SSS_ClampSLDist`, บรรทัด 2972 และ `GetRejectionSL` 4030)
   ถ้าระยะ < `InpM15S_ATR_MinPts` จะใช้ `ATR×2` แทน ซึ่งมักเล็กกว่า $7 และอาจ **แคบกว่า** SL ตามโครงสร้างเดิม
   (SL ไปอยู่ในกรอบ pullback) — ถ้าตั้งใจให้เป็นขั้นต่ำควรเป็น `MathMax(dist, MathMax(minSL, atr*mult))`
3. **หน่วย ATR gate ไม่ตรงกับ input อื่น** (`SSS_GetSignal` บรรทัด 2778, `SSS_GetSignalAt` 4076) ใช้ `Point*10`
   ขณะที่ SL/buffer ใช้ `Point` — บนโบรก 3 ทศนิยม ATR ต้องอยู่ระหว่าง $2–$80; ถ้าย้ายไปโบรก 2 ทศนิยม
   ขั้นต่ำจะกลายเป็น $20 และ SL ขั้นต่ำ $70 (EA แทบไม่เข้าออเดอร์/เสี่ยงเกิน)
4. **RR ไม่ถูกจำกัดด้วย session limit / spike filter / cooldown** — เช็คแค่จำนวนออเดอร์ SSS ที่เปิดอยู่ < `InpMaxOrders`
   จึงเปิดซ้ำใน session ที่ "ครบแล้ว" ได้ถ้าชุดก่อนปิดไปแล้ว
5. `SSS_IsBlockedBySRZone()` มี side effect (ตั้ง persisted block) แต่ถูกเรียกจาก RR กับ breakout ย้อนหลังทุกแท่ง
   ทำให้ breakout เก่าตั้ง block ใหม่ได้แม้ main path ไม่ได้ถูกบล็อก
6. Double-Reject Close ประเมินแท่ง `shift 1` ทันทีใน tick แรกหลังเปิดออเดอร์ = แท่ง **ก่อน** เข้าออเดอร์ (เล็กน้อย)
7. Guard 3 (`Guard_VolOK`) เทียบ ATR(H1) แท่ง 1 กับ ATR แท่งที่ 24 ค่าเดียว ไม่ใช่ค่าเฉลี่ย 24 ชม. ตามชื่อตัวแปร
