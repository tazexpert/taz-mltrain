# ผลตรวจโค้ด ML + ผลบนข้อมูลจริง (2026-11-24)

ข้อมูล: `taz_ml/XAUUSD_M5.csv` (2024-01-30 → 2026-11-24, 200k แท่ง), `taz_ml/XAUUSD_M15.csv` (2018-06 → 2026-11, 200k แท่ง)
เวลาในไฟล์เป็น UTC (ช่วงพักตลาด ~21:00) ซึ่งตรงกับ server time ที่ session ของ EA อ้างอิง

## ปัญหาที่พบในข้อมูล
- **`volume` ไม่ใช่ tick volume** — M5 มีค่า 1–5, M15 ≤ 15 (= จำนวนแท่ง M1) → `vol_ratio` มีค่า 1.000 ± 0.002 ไม่มีข้อมูลเลย
  แต่ยังถูกใช้เป็นฟีเจอร์ในทุกสคริปต์ (และใน `server.py`) ควร export tick volume จริงหรือตัดฟีเจอร์นี้ทิ้ง
- มีแท่งวันอาทิตย์ (2,924 แท่ง M15) → `groupby(date)` ใน Asian breakout สร้าง "วัน" สั้นๆ ที่ทำให้ Daily ATR/EMA เพี้ยน

## ผลรันสคริปต์ของคุณ

| สคริปต์ | ผล OOS | สรุป |
|---|---|---|
| `train_model.py` (multiclass ทุกแท่ง) | BUY precision 0.31 / SELL 0.33, recall 0.02, ไม่มีสัญญาณ ≥ 65% | ไม่มี edge (RR 2:1 ต้องได้ > 33% + spread) |
| `train_meta_model.py` (BB rejection) | baseline -0.04R; ที่ thr 0.35 = +0.10R (155 เทรด) | threshold ถูกเลือกโดยดูชุด test → ตัวเลข +0.10R เชื่อไม่ได้ |
| `walk_forward.py` (BB rejection) | 1,041 เทรด, **-0.07R, -70R** | ไม่มี edge ทุก fold |
| `train_asian_meta_wf.py` | base +3.1R → AI **+29.5R** (440/652 เทรด, +0.07R) | มีสัญญาณ edge แต่ยังเปราะ (ดูด้านล่าง) |
| `test_asian_breakout.py` | SL 1.0x: +17.6R/782 เทรด (+0.02R) | ใกล้เสมอตัว ปี 2023 -19.9R |

### ทดสอบความทนทาน Asian Breakout + AI
- เปลี่ยน seed 10 ครั้ง (เปิด bagging จริง): +19.7 ถึง +31.1R — สม่ำเสมอ
- keep top 50 / 65 / 80 %: +37.8 / +29.5 / +7.6R
- จำนวน fold 4 / 5 / 6 / 8: +22.0 / +29.5 / **+4.5 (แย่กว่า base +8.1)** / +32.8R
- ตัด `vol_ratio` (ฟีเจอร์ว่าง) ออก: +20.2R
- สุ่มเลือก 440 จาก 652 เทรด OOS 5,000 ครั้ง: เฉลี่ย +2.0R, P(≥ +29.5R) = 0.014
- สรุป: น่าจะมี edge เล็กน้อย (~+0.05–0.07R/เทรดก่อนค่าคอม) แต่ทดลองหลายแบบแล้ว (data snooping) ควรเก็บข้อมูล forward test ก่อนใช้เงินจริง

## บั๊ก/ความเสี่ยงในโค้ด
1. `subsample=0.8` ใน LightGBM **ไม่ทำงาน** ถ้าไม่ตั้ง `subsample_freq` (> 0) — ทุกสคริปต์
2. `train_meta_model.py`: ตาราง threshold คำนวณบนชุด test แล้วเลือกค่าที่ดีที่สุด = ใช้ test ในการเลือกพารามิเตอร์
3. `walk_forward.py` / `train_meta_model.py`: label มองไปข้างหน้า 32 แท่ง แต่ไม่มี purge/embargo ระหว่าง train→val→test
   (แถวท้าย train ใช้ราคาที่อยู่ในช่วง val/test) และสัญญาณติดกันมี label ซ้อนกัน
4. ชุด validation ใช้ทั้ง early stopping **และ** คำนวณ dynamic threshold → threshold มองโลกในแง่ดี;
   fold แรกของ Asian มี val แค่ ~26 เทรด ทำให้ threshold แกว่งมาก
5. `grid_search_rr.py` ตัด in-sample ที่ 85% ของ *แท่ง* ขณะที่ `train_meta_model.py` ตัดที่ 85% ของ *สัญญาณ* → ขอบเขตไม่ตรงกัน
   in-sample ของ grid search อาจคาบเกี่ยวชุด test; ฟิลเตอร์ที่ได้ (hour 1–11, vol_regime < 1.1) ก็ไม่ได้ถูกใช้ใน `walk_forward.py`
6. `grid_search_rr.py` / `analyze_filters.py` นับเทรดที่หมดเวลาเป็น -1R เต็ม (ควรคิดตามราคาปิดจริงแบบใน Asian script)
7. `train_model.py`: 12 แถวท้ายได้ label 0 (HOLD) ทั้งที่ยังไม่รู้ผล — ควรเป็น NaN แล้ว drop
8. `server.py` ไม่ตรวจว่า feature ที่ EA ส่งมาคำนวณแบบเดียวกับตอนเทรน (`vol_ratio`, `entry_hour` ต้องเป็นเวลา UTC เหมือน CSV)
9. ไฟล์ `__pycache__/*.pyc` ถูก commit (ไม่ควรอยู่ใน git)
10. `eval_set=` ใน LightGBM รุ่นใหม่ deprecated (ยังทำงาน)

## EA SSS v7.22 บนข้อมูลจริง (`python -m taz_ml.train --csv taz_ml/XAUUSD_M5.csv`)
- 2,286 ออเดอร์: win 53.7%, **-0.0125R/ออเดอร์, PF 0.97, -28.6R**, Max DD -72R
  - Pullback entry: +0.006R (1,728) · Reject+Rebound: **-0.069R (558, -38.6R)**
- S/R block ถูก Reject+Rebound เปิดแทนในแท่งเดียวกัน **426 จาก 622 ครั้ง**
- SL หลัง clamp ต่ำกว่า $7: 2,028 จาก 5,446 สัญญาณ (ATR×2 fallback)
- โมเดล ML กรองสัญญาณ SSS: OOS AUC **0.507** → ไม่มี edge ไม่แนะนำให้ใช้

What-if (`python tools/whatif.py taz_ml/XAUUSD_M5.csv`):

| variant | n | avg R | total R | PF |
|---|---|---|---|---|
| EA v7.22 | 2286 | -0.0125 | -28.6 | 0.971 |
| ไม่เปิดออเดอร์ RR (เก็บ S/R filter) | 2212 | -0.0157 | -34.8 | 0.964 |
| ปิด RR ทั้งหมด (S/R filter หายด้วย) | 2276 | -0.0183 | -41.6 | 0.958 |
| บังคับ SL ขั้นต่ำ $7 | 2286 | -0.0077 | -17.5 | 0.982 |
| ไม่เปิด RR + SL ขั้นต่ำ $7 | 2211 | -0.0098 | -21.7 | 0.977 |

→ การแก้บั๊กไม่ทำให้ SSS มีกำไร: เมื่อตัดออเดอร์ RR ออก slot ใน session ถูกแทนด้วยสัญญาณอื่นที่ผลพอๆ กัน
ตัวกลยุทธ์ SSS (หน่วย R ไม่รวมค่าคอม/การทบ lot) อยู่ราวเสมอตัวถึงติดลบเล็กน้อยในช่วง 2024–2026
