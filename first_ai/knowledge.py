"""Bounded text corpus, local lexical retrieval, optional NIM dense retrieval."""
import hashlib
import json
import math
import re
import unicodedata
from .storage import AppError, no_symlinks, read_text


def tokens(text):
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", normalized)
    normalized = re.sub("[أإآ]", "ا", normalized).replace("ى", "ي")
    words = re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
    result = set()
    for word in words:
        if len(word) <= 1:
            continue
        # Retain the literal spelling too: this is bounded lexical expansion,
        # not a morphological parser or a reason to rewrite source text/IDs.
        variants = {word}
        if len(word) > 5 and word.startswith(("وال", "فال", "بال", "كال")):
            variants.add(word[1:])
        if len(word) > 4 and word.startswith("لل"):
            variants.add("ال" + word[2:])
        variants.update(form[2:] for form in tuple(variants) if form.startswith("ال") and len(form) > 4)
        result.update(variants)
    return result


class Corpus:
    def __init__(self, directory):
        self.root = no_symlinks(directory)
        if not self.root.is_dir():
            raise AppError("مجلد المعرفة غير موجود.")
        self.chunks = []
        total = 0
        for path in sorted(self.root.iterdir()):
            if path.name.startswith(".") or path.suffix.lower() not in (".md", ".txt"):
                continue
            text = read_text(path)
            total += len(text.encode())
            if total > 2 * 1024 * 1024:
                raise AppError("ملفات المعرفة تجاوزت حد النسخة التعليمية.")
            lines = text.splitlines()
            blocks, buffer, start, size = [], [], 1, 0
            for number, line in enumerate(lines, 1):
                if len(line) > 1200:
                    raise AppError("أحد أسطر المعرفة طويل جدًا؛ قسمه إلى أسطر أقصر.")
                if buffer and size + len(line) > 1200:
                    blocks.append((start, number - 1, "\n".join(buffer)))
                    buffer, start, size = [], number, 0
                buffer.append(line)
                size += len(line) + 1
            if buffer:
                blocks.append((start, len(lines), "\n".join(buffer)))
            for start, end, body in blocks:
                if body.strip():
                    digest = hashlib.sha256((path.name + str(start) + body).encode()).hexdigest()[:16]
                    self.chunks.append({"id": "S" + digest, "file": path.name,
                                        "start": start, "end": end, "text": body})
            if len(self.chunks) > 200:
                raise AppError("الحد الحالي مئتا مقطع؛ قلل الملفات قبل الفهرسة.")
        self.by_id = {c["id"]: c for c in self.chunks}
        self.fingerprint = hashlib.sha256(json.dumps(self.chunks, sort_keys=True).encode()).hexdigest()

    def search(self, query, client=None, index=None, limit=4):
        if index is not None:
            if not isinstance(index, dict) or client is None or index.get("fingerprint") != self.fingerprint or index.get("model") != client.cfg["embedding_model"]:
                raise AppError("الفهرس قديم أو نموذجه مختلف؛ أعد بناءه.")
            query_vector = client.embed([query], "query")[0]
            vectors = index.get("vectors", [])
            if len(vectors) != len(self.chunks):
                raise AppError("عدد مقاطع الفهرس غير صالح؛ أعد بناءه.")
            scores = [cosine(query_vector, vector) for vector in vectors]
        else:
            query_terms = tokens(query)
            scores = []
            for chunk in self.chunks:
                terms = tokens(chunk["text"])
                scores.append(len(query_terms & terms) / max(1, math.sqrt(len(terms))))
        matches = sorted(zip(scores, self.chunks), key=lambda x: x[0], reverse=True)
        # Dense scores are relative similarity, not confidence or factual evidence.
        return [{**chunk, "score": round(score, 5)} for score, chunk in matches[:limit] if score > 0]

    def build_index(self, client):
        if not self.chunks:
            raise AppError("لا توجد نصوص صالحة للفهرسة.")
        vectors = []
        for offset in range(0, len(self.chunks), 16):
            vectors.extend(client.embed([c["text"] for c in self.chunks[offset:offset+16]], "passage"))
        dimension = len(vectors[0])
        if any(len(v) != dimension for v in vectors):
            raise AppError("اختلفت أبعاد الفهرس بين الدفعات.")
        return {"version": 1, "fingerprint": self.fingerprint,
                "model": client.cfg["embedding_model"], "vectors": vectors}


def cosine(left, right):
    if not isinstance(right, list) or len(left) != len(right) or any(
        type(x) not in (float, int) or not math.isfinite(x) for x in right
    ):
        raise AppError("متجه بحث غير صالح.")
    norm = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    if norm == 0 or not math.isfinite(norm):
        raise AppError("متجه بحث صفري أو خارج النطاق.")
    return sum(a*b for a, b in zip(left, right)) / norm
