"""
ตั้งบัญชีให้เป็น admin (หรือกลับเป็น member) ในฐานข้อมูล

ใช้:  python backend/scripts/set_admin.py
      - ถามหา Database URL (External URL จาก Render) — พิมพ์แล้วไม่แสดงบนจอ
      - แสดงรายชื่อผู้ใช้ทั้งหมด แล้วถามอีเมลที่จะเปลี่ยน role
ต้องมี:  pip install "psycopg[binary]"
ไม่เก็บ URL / รหัสผ่านไว้ที่ไหน — อย่าเขียน URL ลงไฟล์ที่ commit ขึ้น Git
"""
import getpass
import os
import sys

try:
    import psycopg
except ImportError:
    sys.exit('ยังไม่ได้ติดตั้ง psycopg — รัน:  pip install "psycopg[binary]"')


def normalise(url: str) -> str:
    url = url.strip().strip('"').strip("'")
    for prefix in ("postgresql+asyncpg://", "postgres+asyncpg://", "postgres://"):
        if url.startswith(prefix):
            url = "postgresql://" + url[len(prefix):]
    return url


def main() -> None:
    url = os.environ.get("DB_URL") or getpass.getpass("วาง External Database URL แล้วกด Enter (ตัวอักษรจะไม่แสดง): ")
    url = normalise(url)
    if not url.startswith("postgresql://"):
        sys.exit("URL ต้องขึ้นต้นด้วย postgresql://")

    with psycopg.connect(url, connect_timeout=20) as conn:
        rows = conn.execute("SELECT email, role FROM users ORDER BY created_at").fetchall()
        print(f"\nผู้ใช้ทั้งหมด {len(rows)} คน:")
        for email, role in rows:
            print(f"  {role:<8} {email}")

        email = input("\nอีเมลที่จะตั้งเป็น admin (เว้นว่าง = ยกเลิก): ").strip()
        if not email:
            print("ยกเลิก ไม่ได้เปลี่ยนอะไร")
            return
        role = "member" if input("พิมพ์ m ถ้าจะเปลี่ยนกลับเป็น member, กด Enter = admin: ").strip().lower() == "m" else "admin"

        cur = conn.execute("UPDATE users SET role = %s::user_role WHERE lower(email) = lower(%s)", (role, email))
        conn.commit()
        if cur.rowcount:
            print(f"\nเปลี่ยน {email} เป็น {role} แล้ว — logout แล้ว login ใหม่ในเว็บ")
        else:
            print(f"\nไม่พบอีเมล {email} ในตาราง users (เช็กตัวสะกดจากรายชื่อด้านบน)")


if __name__ == "__main__":
    main()
