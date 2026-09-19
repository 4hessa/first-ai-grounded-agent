"""Private local state. The local operator and project directory are trusted."""
import json
import os
import re
import stat
import tempfile
from pathlib import Path


class AppError(Exception):
    """An error whose message is safe to display without remote response bodies."""

    def __init__(self, message, *, code="application_error", http_status=None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status

    def details(self):
        result = {"error": str(self), "error_code": self.code}
        if self.http_status is not None:
            result["http_status"] = self.http_status
        return result


def no_symlinks(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise AppError("المسار يحتوي على رابط رمزي؛ اختر مجلدًا عاديًا.")
    return path


def read_text(path, limit=262144):
    path = no_symlinks(path)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise AppError("نوع الملف أو حجمه غير مسموح.")
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise AppError("الملف أكبر من الحد المسموح.")
        return data.decode("utf-8")
    except (OSError, UnicodeError):
        raise AppError("تعذرت قراءة الملف المحلي؛ تحقق من وجوده وترميزه.") from None


def read_json(path, limit=262144):
    try:
        return json.loads(read_text(path, limit))
    except (ValueError, RecursionError):
        raise AppError("ملف البيانات المحلي غير صالح.") from None


class Store:
    def __init__(self, root):
        self.root = no_symlinks(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def path(self, name):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.json", name):
            raise AppError("اسم ملف الحالة غير صالح.")
        return no_symlinks(self.root / name)

    def read(self, name, default=None):
        path = self.path(name)
        return read_json(path, 32 * 1024 * 1024) if path.exists() else default

    def write(self, name, value):
        path = self.path(name)
        fd, temporary = tempfile.mkstemp(prefix="save-", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def create_once(self, name, value):
        """Claim a single resume attempt before sending any request.

        A partial file after interruption still occupies the claim. Never erase
        it and risk sending the same bounded attempt from another process.
        """
        path = self.path(name)
        data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise AppError("سبق تسجيل محاولة استكمال لهذا البحث؛ لم يُرسل طلب جديد.") from None
        except OSError:
            raise AppError("تعذر تسجيل محاولة الاستكمال محليًا؛ لم يُرسل طلب جديد.") from None
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            raise AppError("تعذر إكمال تسجيل محاولة الاستكمال؛ لم يُرسل طلب جديد. احتُفظ بسجل الحجز.") from None

    def notes(self):
        notes = self.read("notes.json", [])
        if not isinstance(notes, list) or len(notes) > 30 or any(
            not isinstance(n, str) or len(n) > 500 for n in notes
        ):
            raise AppError("ملف الملاحظات غير صالح.")
        return notes

    def add_note(self, text):
        if not text.strip() or len(text) > 500:
            raise AppError("الملاحظة يجب أن تكون بين حرف واحد و٥٠٠ حرف.")
        if re.search(r"nvapi-|BEGIN .*PRIVATE KEY|Bearer\s+\S+", text, re.I):
            raise AppError("يبدو النص كبيانات اعتماد. لا تحفظ الأسرار في ذاكرة المساعد.")
        notes = self.notes()
        if len(notes) >= 30:
            raise AppError("وصلت الملاحظات إلى الحد؛ احذف ملاحظة قديمة أولًا.")
        notes.append(text.strip())
        self.write("notes.json", notes)

    def delete_note(self, number):
        notes = self.notes()
        if number < 1 or number > len(notes):
            raise AppError("رقم الملاحظة غير موجود.")
        del notes[number - 1]
        self.write("notes.json", notes)
