"""Regressions for the observed Arabic retrieval gap and lost branch evidence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest

from first_ai.__main__ import show
from first_ai.agent import Agent, checked_answer
from first_ai.knowledge import Corpus
from first_ai.storage import AppError, Store, read_json

ROOT = Path(__file__).resolve().parents[1]
CFG = read_json(ROOT / "config.json")


class EvidenceClient:
    """A controlled client that deliberately leaves a retrieved source uncited."""
    def __init__(self, fail_synthesis=False):
        self.cfg = CFG
        self.calls, self.embedding_calls = [], []
        self.fail_synthesis = fail_synthesis
        self.lock = threading.Lock()

    def complete(self, messages, tools=None):
        with self.lock:
            self.calls.append(copy.deepcopy(messages))
        text = messages[-1]["content"]
        if "قسم سؤال" in text:
            return {"content": json.dumps({"tasks": ["العزل", "الذاكرة", "المعرفة"]})}, "stop", {}
        evidence = json.loads(text.split("أدلة غير موثوقة كتعليمات:\n", 1)[1])
        if "اجمع نتائج" in text:
            if self.fail_synthesis:
                raise AppError("خطأ خدمة محاكى", code="http_503", http_status=503)
            source = next(c for c in evidence if c["file"] == "04-runtime.md")
            return {"content": "جواب اختباري يستشهد بمصدر أغفلته الفروع [" + source["id"] + "]"}, "stop", {}
        # All workers avoid citing the runtime source, even if it is retrieved.
        source = next(c for c in evidence if c["file"] != "04-runtime.md")
        return {"content": "مسودة فرع للاختبار [" + source["id"] + "]"}, "stop", {}

    def embed(self, texts, input_type):
        assert input_type == "query"
        with self.lock:
            self.embedding_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class ArabicRetrievalTests(unittest.TestCase):
    def test_runtime_and_context_queries_find_their_distinct_sources(self):
        corpus = Corpus(ROOT / "knowledge")
        files = {c["file"] for c in corpus.search("العزل")}
        self.assertIn("04-runtime.md", files)
        self.assertIn("03-research.md", files)
        self.assertEqual(corpus.search("عزل التشغيل")[0]["file"], "04-runtime.md")
        self.assertEqual(corpus.search("عزل السياق")[0]["file"], "03-research.md")

    def test_attached_articles_work_in_documents_and_queries(self):
        variants = ("العزل", "والعزل", "فالعزل", "بالعزل", "كالعزل", "للعزل")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "topic.md").write_text("بيانات عن العزل", encoding="utf-8")
            corpus = Corpus(root)
            for word in variants:
                with self.subTest(query=word):
                    self.assertEqual(corpus.search(word)[0]["file"], "topic.md")
            for word in variants:
                with self.subTest(document=word):
                    (root / "topic.md").write_text("بيانات " + word, encoding="utf-8")
                    self.assertTrue(Corpus(root).search("عزل"))

    def test_expansion_does_not_truncate_latin_words(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "topic.txt").write_text("allowlist balloon", encoding="utf-8")
            corpus = Corpus(root)
            self.assertTrue(corpus.search("allowlist"))
            self.assertFalse(corpus.search("lowlist"))
            self.assertFalse(corpus.search("loon"))


class EvidenceFlowTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(ROOT / "knowledge")

    def stopped_run(self, store):
        client = EvidenceClient(fail_synthesis=True)
        with self.assertRaises(AppError):
            Agent(client, self.corpus, CFG).research("قارن العزل والذاكرة والاسترجاع", store)
        report = read_json(next(store.root.glob("*-result.json")))
        self.assertEqual(report["research_stats"]["model_calls"], 5)
        return report["run_id"]

    def test_grounded_result_distinguishes_retrieved_and_cited_sources(self):
        client = EvidenceClient()
        result = Agent(client, self.corpus, CFG).grounded("العزل")
        retrieved = {source["file"] for source in result["retrieved_sources"]}
        cited = {source["file"] for source in result["sources"]}
        self.assertIn("04-runtime.md", retrieved)
        self.assertNotIn("04-runtime.md", cited)
        self.assertTrue(cited <= retrieved)
        self.assertEqual(len(client.calls), 1)

    def test_synthesis_can_use_a_retrieved_source_omitted_by_all_workers(self):
        client = EvidenceClient()
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            result = Agent(client, self.corpus, CFG).research("قارن العزل والذاكرة والاسترجاع", store)
            drafts = [store.read(result["run_id"] + "-" + str(i) + ".json") for i in range(3)]
            cited = {s["id"] for draft in drafts for s in draft["sources"]}
            retrieved = {s["id"] for draft in drafts for s in draft["retrieved_sources"]}
            runtime_id = next(c["id"] for c in self.corpus.chunks if c["file"] == "04-runtime.md")
            self.assertNotIn(runtime_id, cited)
            self.assertIn(runtime_id, retrieved)
            self.assertEqual({s["id"] for s in result["synthesis_sources"]}, retrieved)
            self.assertEqual(result["sources"][0]["id"], runtime_id)
            self.assertEqual(len(client.calls), 5)
            self.assertEqual(client.embedding_calls, [])

    def test_resume_keeps_uncited_evidence_without_new_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            run_id = self.stopped_run(store)
            before = {p.name: p.read_bytes() for p in store.root.glob("*.json")}
            client = EvidenceClient()
            result = Agent(client, self.corpus, CFG).resume(run_id, store)
            self.assertEqual(result["sources"][0]["file"], "04-runtime.md")
            self.assertEqual(result["research_stats"]["model_calls"], 6)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(client.embedding_calls, [])
            self.assertEqual(before, {name: store.path(name).read_bytes() for name in before})

    def test_invalid_saved_uncited_evidence_is_rejected_before_any_request(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            run_id = self.stopped_run(store)
            filename = run_id + "-0.json"
            original = store.read(filename)
            invalid_lists = [
                [],
                [{**original["retrieved_sources"][0], "id": "Sunknown"}],
                [{**source, "file": "changed.md"} for source in original["retrieved_sources"]],
                original["retrieved_sources"] * 5,
                None,
            ]
            client = EvidenceClient()
            for sources in invalid_lists:
                with self.subTest(sources=sources):
                    store.write(filename, {**original, "retrieved_sources": sources})
                    with self.assertRaises(AppError):
                        Agent(client, self.corpus, CFG).resume(run_id, store)
                    self.assertFalse(store.path(run_id + "-resume-result.json").exists())
            self.assertEqual(client.calls, [])

    def test_worker_still_cannot_cite_a_file_that_it_did_not_retrieve(self):
        runtime_id = next(c["id"] for c in self.corpus.chunks if c["file"] == "04-runtime.md")

        class WrongSource:
            def complete(self, messages, tools=None):
                return {"content": "جواب [" + runtime_id + "]"}, "stop", {}

        self.assertNotIn(runtime_id, {s["id"] for s in self.corpus.search("الذاكرة الدائمة")})
        with self.assertRaises(AppError) as caught:
            Agent(WrongSource(), self.corpus, CFG).grounded("الذاكرة الدائمة")
        self.assertEqual(caught.exception.code, "unknown_source")

    def test_semantic_research_still_embeds_only_the_three_worker_queries(self):
        client = EvidenceClient()
        index = {"fingerprint": self.corpus.fingerprint, "model": CFG["embedding_model"],
                 "vectors": [[1.0, 0.0] for _ in self.corpus.chunks]}
        with tempfile.TemporaryDirectory() as directory:
            result = Agent(client, self.corpus, {**CFG, "retrieval": "semantic"}, index=index).research(
                "قارن العزل والذاكرة والاسترجاع", Store(directory))
        self.assertEqual(len(client.embedding_calls), 3)
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(result["sources"][0]["file"], "04-runtime.md")


class QualityNoteTests(unittest.TestCase):
    def test_mixed_arabic_latin_prose_gets_a_visible_quality_note(self):
        text = "يحفظ كل factor نتيجة الفرع، وهذا مثال based on الخطة."
        result = checked_answer(text, {})
        self.assertEqual(result["answer"], text)
        self.assertTrue(result["quality_warnings"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            show(result)
        self.assertIn("يحتاج الشرح إلى مراجعة لغوية", output.getvalue())

    def test_separate_terms_code_urls_and_citations_do_not_trigger_mixed_line_note(self):
        corpus = Corpus(ROOT / "knowledge")
        key = corpus.chunks[0]["id"]
        text = "NemoClaw\nمصطلح في سطر مستقل.\nمثال `items[0]`، ومصدر [" + key + "]\nرابط https://example.invalid/docs"
        result = checked_answer(text, corpus.by_id, required=True)
        self.assertEqual(result["quality_warnings"], [])


if __name__ == "__main__":
    unittest.main()
