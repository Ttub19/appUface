/**
 * Reference data for the 5 face shapes the mock classifier can return.
 * Each entry carries the style recommendations shown on the result screen
 * and used to pre-select a matching AR accessory style.
 * @type {Array<Object>}
 */
export const faceData = [
    { id: "oval", name: "หน้ารูปไข่ (Oval)", desc: "สัดส่วนสมดุลที่สุด ความกว้างของโหนกแก้มใกล้เคียงกับความยาวหน้า คางมนเรียว", hair: { male: "Pompadour หรือ Undercut เปิดหน้าผาก", female: "สไลด์เลเยอร์ยาว หรือบ๊อบเท" }, glass: "แว่นทรงเหลี่ยมช่วยเพิ่มมิติ", glassStyle: "wayfarer", jewelry: "ต่างหูทรงรี หรือแบบเรียบง่าย", beard: "เคราสั้นตกแต่งทรงให้คมชัด", makeup: "เน้นปัดแก้มตามโหนกแก้ม" },
    { id: "round", name: "หน้ากลม (Round)", desc: "ความกว้างและความยาวใกล้เคียงกัน เส้นกรามโค้งมน ไม่มีมุมชัดเจน", hair: { male: "Quiff หรือ Faux Hawk เพิ่มความสูง", female: "บ๊อบยาว (Long Bob) หรือทรงผมยาวแสกข้าง" }, glass: "แว่นทรงสี่เหลี่ยมผืนผ้าเพื่อพรางหน้าให้ยาวขึ้น", glassStyle: "classic", jewelry: "ต่างหูแนวดิ่ง ทรงหยดน้ำ", beard: "เคราทรงแพะ (Goatee) ช่วยยืดปลายคาง", makeup: "เฉดดิ้งข้างแก้มให้ดูเรียว" },
    { id: "square", name: "หน้าเหลี่ยม (Square)", desc: "กรามและหน้าผากกว้างใกล้เคียงกัน เส้นขากรรไกรเป็นมุมชัดเจน", hair: { male: "ทรงผมสั้นไถข้างเล็กน้อยแต่ข้างบนหนา", female: "ผมดัดลอนใหญ่ หรือผมยาวแสกข้างพรางกราม" }, glass: "แว่นทรงกลม หรือทรงหยดน้ำ (Aviator)", glassStyle: "aviator", jewelry: "ต่างหูทรงกลม หรือห่วง", beard: "เคราเต็ม (Full Beard) แต่ตัดขอบให้มน", makeup: "ไฮไลท์กึ่งกลางหน้าผากและคาง" },
    { id: "heart", name: "หน้ารูปหัวใจ (Heart)", desc: "หน้าผากกว้างเด่น คางแหลมเรียว คล้ายรูปหัวใจกลับหัว", hair: { male: "Undercut ที่ไม่ไถสั้นเกินไป หรือทรงผมแสกข้าง", female: "หน้าม้าปัดข้าง หรือผมลอนสไลด์ช่วงล่าง" }, glass: "แว่นทรงฐานกว้าง หรือทรง Cat-eye", glassStyle: "cat-eye", jewelry: "ต่างหูทรงสามเหลี่ยมหัวกลับ", beard: "เคราหนาช่วงคางเพื่อเพิ่มวอลลุ่ม", makeup: "เฉดดิ้งขมับพรางหน้าผากกว้าง" },
    { id: "oblong", name: "หน้ายาว (Oblong)", desc: "ความยาวใบหน้ามากกว่าความกว้างอย่างชัดเจน หน้าผากและคางค่อนข้างยาว", hair: { male: "ทรงผมแสกข้าง (Side Part) ไม่ควรเน้นตั้งสูง", female: "หน้าม้าตัดตรง หรือผมดัดลอนพองออกด้านข้าง" }, glass: "แว่นทรงกว้างหรือเลนส์ขนาดใหญ่ (Oversized)", glassStyle: "oversized", jewelry: "ต่างหูขนาดใหญ่ทรงกลม", beard: "เคราที่เลี้ยงยาวออกด้านข้างแก้ม", makeup: "ปัดบลัชออนแนวนอนตัดความยาว" }
];
