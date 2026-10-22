const META={pending:["รอยืนยัน","warning"],confirmed:["ยืนยันแล้ว","success"],rejected:["ร้านปฏิเสธ","danger"],cancelled:["ยกเลิกแล้ว","neutral"],completed:["เสร็จสิ้น","info"]};
export const escapeHtml=v=>String(v??"").replace(/[&<>'"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[c]);
export function parseTags(v){const seen=new Set;return String(v??"").split(",").map(x=>x.trim()).filter(x=>{const k=x.toLowerCase();if(!x||seen.has(k))return false;seen.add(k);return true}).slice(0,20)}
export function statusMeta(v){const m=META[v]||[v||"ไม่ทราบสถานะ","neutral"];return{label:m[0],tone:m[1]}}
export function thaiDate(v){const d=new Date(v);return Number.isNaN(d.getTime())?"-":new Intl.DateTimeFormat("th-TH",{dateStyle:"medium",timeStyle:"short",timeZone:"Asia/Bangkok"}).format(d)}
export function mapUrls(lat,lng){lat=Number(lat);lng=Number(lng);if(!Number.isFinite(lat)||!Number.isFinite(lng)||Math.abs(lat)>90||Math.abs(lng)>180)return null;const o=["https:","","www.google.com"].join("/"),p=`${lat},${lng}`;return{embed:`${o}/maps?q=${encodeURIComponent(p)}&z=15&output=embed`,directions:`${o}/maps/dir/?api=1&destination=${encodeURIComponent(p)}`}}
export function localDate(days=1){const d=new Date;d.setDate(d.getDate()+days);return new Date(d-d.getTimezoneOffset()*60000).toISOString().slice(0,10)}
