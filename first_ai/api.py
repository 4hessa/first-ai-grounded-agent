"""Fixed HTTPS destinations, bounded requests, and no credential logging."""
import json
import math
import ssl
import urllib.error
import urllib.request
from .storage import AppError


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AppError("أوقف الاتصال: إعادة توجيه غير مسموحة.")


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise AppError("إعدادات المشروع غير صالحة.")
    if cfg.get("provider") not in ("nvidia", "openshell"):
        raise AppError("اختر أحد مزودي الاتصال الموثقين في الدليل.")
    if cfg.get("retrieval") not in ("keyword", "semantic"):
        raise AppError("طريقة البحث غير صالحة.")
    for name, lo, hi in (
        ("max_model_calls", 1, 8), ("max_output_tokens", 64, 8192),
        ("timeout_seconds", 1, 90), ("max_question_chars", 1, 8000),
        ("max_context_chars", 4000, 60000),
    ):
        value = cfg.get(name)
        if type(value) is not int or not lo <= value <= hi:
            raise AppError("أحد حدود الإعدادات مفقود أو خارج النطاق المسموح.")
    for name in ("model", "embedding_model"):
        value = cfg.get(name)
        if not isinstance(value, str) or not value or len(value) > 150:
            raise AppError("اسم النموذج غير صالح.")
    for name, hi in (("temperature", 2), ("top_p", 1)):
        value = cfg.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= hi:
            raise AppError("إعدادات توليد النص غير صالحة.")
    extra = cfg.get("chat_template_kwargs", {})
    if not isinstance(extra, dict) or set(extra) - {"enable_thinking"} or any(
        type(v) is not bool for v in extra.values()
    ):
        raise AppError("إعداد قالب المحادثة غير صالح.")
    return cfg


class NvidiaClient:
    def __init__(self, cfg, key=None):
        self.cfg = validate_config(cfg)
        self.provider = cfg["provider"]
        if self.provider == "nvidia" and (not key or not key.strip()):
            raise AppError("مفتاح الخدمة غير موجود؛ أدخله محليًا عند التشغيل.")
        # OpenShell holds the upstream credential; never forward the host key.
        self._key = key.strip() if self.provider == "nvidia" else "openshell"
        self._base = (
            "https://integrate.api.nvidia.com/v1" if self.provider == "nvidia"
            else "https://inference.local/v1"
        )

    def _post(self, endpoint, payload):
        if endpoint not in ("chat/completions", "embeddings"):
            raise AppError("مسار الخدمة غير مسموح.")
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if len(body) > 220000:
            raise AppError("حجم الطلب تجاوز الحد المسموح.")
        request = urllib.request.Request(
            self._base + "/" + endpoint, data=body, method="POST",
            headers={"Authorization": "Bearer " + self._key,
                     "Content-Type": "application/json", "Accept": "application/json"},
        )
        # Honor the operating system's trusted CA store; never disable TLS checks.
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl.create_default_context()), NoRedirect()
        )
        try:
            with opener.open(request, timeout=self.cfg["timeout_seconds"]) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise AppError("استجابة الخدمة أكبر من الحد.")
            data = json.loads(raw)
            if not isinstance(data, dict) or "error" in data:
                raise AppError("الخدمة أعادت استجابة غير صالحة.")
            return data
        except urllib.error.HTTPError as exc:
            messages = {
                400: "الطلب أو قدرة النموذج غير متوافقة؛ راجع إعداداته.",
                401: "تعذر اعتماد مفتاح الخدمة.",
                403: "الحساب أو سياسة الشبكة لا تسمح بهذا الطلب.",
                404: "المسار أو النموذج غير متاح لهذا الحساب.",
                408: "أبلغت الخدمة عن انتهاء مهلة الطلب.",
                413: "رفضت الخدمة حجم الطلب.",
                429: "تم بلوغ حد الاستخدام؛ أعد المحاولة لاحقًا.",
                500: "أبلغت الخدمة عن خطأ داخلي.",
                502: "أبلغت البوابة عن استجابة غير صالحة من الخدمة التي تتصل بها.",
                503: "أبلغت الخدمة بأنها غير متاحة لهذا الطلب.",
                504: "أبلغت البوابة عن انتهاء مهلة انتظار الخدمة التي تتصل بها.",
            }
            status = exc.code
            # Only the numeric status is retained. Never expose remote bodies,
            # reason phrases, headers, request payloads, or credentials.
            exc.close()
            message = messages.get(status, "رفضت الخدمة الطلب؛ لا يتوفر تفسير محلي لهذا الرمز.")
            raise AppError(message + "\nرمز استجابة الخدمة: " + str(status),
                           code="http_" + str(status), http_status=status) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise AppError("تعذر الاتصال أو انتهت مهلة الطلب؛ افحص الاتصال والشهادة والمسار.",
                           code="connection_error") from None
        except (ValueError, RecursionError):
            raise AppError("لم تُرجع الخدمة بيانات صالحة.") from None

    def complete(self, messages, tools=None):
        cfg = self.cfg
        payload = {
            "model": cfg["model"], "messages": messages, "stream": False,
            "max_tokens": cfg["max_output_tokens"],
            "temperature": cfg["temperature"], "top_p": cfg["top_p"],
        }
        if cfg.get("chat_template_kwargs"):
            payload["chat_template_kwargs"] = cfg["chat_template_kwargs"]
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        data = self._post("chat/completions", payload)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError()
            return message, choice.get("finish_reason"), data.get("usage", {})
        except (KeyError, TypeError, IndexError):
            raise AppError("استجابة المحادثة لا تتبع البنية المتوقعة.") from None

    def embed(self, texts, input_type):
        if self.provider != "nvidia":
            raise AppError("مسار التضمين المعزول لم يُضبط بعد. استخدم البحث المحلي في هذه المرحلة.")
        if input_type not in ("query", "passage") or not 1 <= len(texts) <= 16:
            raise AppError("دفعة التضمين غير صالحة.")
        data = self._post("embeddings", {
            "model": self.cfg["embedding_model"], "input": texts,
            "input_type": input_type, "encoding_format": "float", "truncate": "NONE",
        })
        try:
            rows = sorted(data["data"], key=lambda row: row["index"])
            if [row["index"] for row in rows] != list(range(len(texts))):
                raise ValueError()
            vectors = [row["embedding"] for row in rows]
            dimension = len(vectors[0])
            if not 1 <= dimension <= 8192 or any(
                not isinstance(v, list) or len(v) != dimension or any(
                    type(x) not in (float, int) or not math.isfinite(x) for x in v
                ) or sum(x*x for x in v) <= 0 for v in vectors
            ):
                raise ValueError()
            return vectors
        except (KeyError, TypeError, ValueError, IndexError):
            raise AppError("أبعاد أو قيم متجهات البحث غير صالحة.") from None
