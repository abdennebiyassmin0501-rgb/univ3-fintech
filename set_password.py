"""
set_password.py — إنشاء أو تغيير حساب المالك الوحيد للمنصة.

الاستعمال:
    python set_password.py                 (يسألك عن الاسم وكلمة المرور)
    python set_password.py univ3 كلمةالسر   (مباشرة)

لا تُحفَظ كلمة المرور نفسها، بل «بصمتها» المحسوبة بـ PBKDF2-SHA256 مع ملح عشوائي
و200 000 تكرار. حتى من يقرأ auth.json لا يستطيع استرجاع كلمة المرور منه مباشرة.
"""
import getpass
import hashlib
import json
import secrets
import sys
from pathlib import Path

AUTH_PATH = Path(__file__).parent / "auth.json"
ITERATIONS = 200_000


def make_record(username, password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return {"username": username.strip(), "salt": salt.hex(), "iterations": ITERATIONS, "hash": digest.hex()}


def main():
    if len(sys.argv) == 3:
        username, password = sys.argv[1], sys.argv[2]
    else:
        username = input("اسم المستخدم: ").strip()
        password = getpass.getpass("كلمة المرور: ")
        if password != getpass.getpass("أعد كتابة كلمة المرور: "):
            print("كلمتا المرور غير متطابقتين. لم يتغير شيء.")
            return
    if not username or len(password) < 8:
        print("اسم المستخدم مطلوب، وكلمة المرور يجب أن تكون 8 أحرف على الأقل.")
        return
    rec = make_record(username, password)
    AUTH_PATH.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"تم حفظ الحساب «{username}» في {AUTH_PATH.name}. أعد تشغيل التطبيق.")
    print("\nللاستضافة على Streamlit Cloud، الصق هذا في Secrets:\n")
    print("[auth]")
    for k, v in rec.items():
        print(f'{k} = {v}' if isinstance(v, int) else f'{k} = "{v}"')


if __name__ == "__main__":
    main()
