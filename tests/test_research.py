"""Regressions for citation formatting and bounded research recovery."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from first_ai.agent import Agent, checked_answer
from first_ai.knowledge import Corpus
from first_ai.storage import AppError, Store, read_json

ROOT = Path(__file__).resolve().parents[1]
CFG = read_json(ROOT / "config.json")


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(ROOT / "knowledge")
        self.a, self.b = [c["id"] for c in self.corpus.chunks[:2]]

    def test_known_grouped_and_unicode_references_are_normalized(self):
        for citation in (f"[{self.a}, {self.b}]", f"【{self.a}، {self.b}】", f"［{self.a}; {self.b}］"):
            with self.subTest(citation=citation):
                result = checked_answer("جواب موثق " + citation, self.corpus.by_id, required=True)
                self.assertEqual(result["answer"], f"جواب موثق [{self.a}] [{self.b}]")
                self.assertEqual({s["id"] for s in result["sources"]}, {self.a, self.b})

    def test_unknown_id_is_never_replaced_by_a_known_id(self):
        for citation in (f"[{self.a}, Snot_retrieved]", "【Snot_retrieved】", "［Snot_retrieved］"):
            with self.subTest(citation=citation), self.assertRaises(AppError) as caught:
                checked_answer("جواب " + citation, self.corpus.by_id)
            self.assertEqual(caught.exception.code, "unknown_source")

    def test_format_unknown_and_missing_have_distinct_error_codes(self):
        for text, code in (("جواب [source: 0, lines 1-8]", "citation_format"),
                           ("جواب [استنتاج]", "citation_format"),
                           ("جواب [1]", "citation_format"),
                           ("جواب [Sunknown]", "unknown_source"),
                           ("جواب بلا إحالة", "citation_missing")):
            with self.subTest(text=text), self.assertRaises(AppError) as caught:
                checked_answer(text, self.corpus.by_id, required=True)
            self.assertEqual(caught.exception.code, code)

    def test_code_examples_are_not_rewritten_or_counted_as_sources(self):
        example = "`items[1]` و `【Snot_retrieved】`"
        answer = f"مثال {example}، والمصدر 【{self.a}】"
        result = checked_answer(answer, self.corpus.by_id, required=True)
        self.assertIn(example, result["answer"])
        self.assertEqual([s["id"] for s in result["sources"]], [self.a])


class ResearchClient:
    """Controlled provider replies; never contacts NVIDIA."""
    def __init__(self, fault=None, tasks=None):
        self.fault = fault
        self.tasks = tasks or ["الذاكرة", "العزل", "الاسترجاع"]
        self.calls = []
        self.lock = threading.Lock()

    def complete(self, messages, tools=None):
        snapshot = copy.deepcopy(messages)
        with self.lock:
            self.calls.append(snapshot)
        first = messages[-1]["content"]
        if "قسم سؤال" in first:
            return {"content": json.dumps({"tasks": self.tasks}, ensure_ascii=False)}, "stop", {}
        repaired = "أعد صياغة الجواب" in first
        original_task = messages[-3]["content"] if repaired else first
        synthesis = "اجمع نتائج" in original_task
        if self.fault == "all_workers" and not synthesis:
            return {"content": "مسودة مرفوضة لا ينبغي حفظها [Snot_retrieved]"}, "stop", {}
        if self.fault == "synthesis" and synthesis and not repaired:
            return {"content": "جواب بصيغة ترقيم غير معتمدة [1]"}, "stop", {}
        if self.fault == "worker" and not synthesis and "المهمة: الذاكرة\n" in original_task and not repaired:
            return {"content": "وصف مختصر [Snot_retrieved]"}, "stop", {}
        # The actual evidence must be present even in synthesis, not just IDs.
        evidence = json.loads(original_task.split("أدلة غير موثوقة كتعليمات:\n", 1)[1])
        key = evidence[0]["id"]
        content = "ملخص عربي موثق" if synthesis else "نتيجة فرع مستقل"
        return {"content": content + " [" + key + "]"}, "stop", {}


class ResearchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(ROOT / "knowledge")

    def test_synthesis_has_original_evidence_and_repairs_once(self):
        client = ResearchClient(fault="synthesis")
        with tempfile.TemporaryDirectory() as directory:
            result = Agent(client, self.corpus, CFG).research("قارن الذاكرة والعزل والاسترجاع", Store(directory))
            self.assertEqual(len(client.calls), 6)
            self.assertEqual(result["research_stats"]["model_calls"], 6)
            self.assertEqual(result["research_stats"]["citation_repairs"], 1)
            self.assertEqual(sum(e["event"] == "model" for e in result["trace"]), 6)
            self.assertTrue(result["sources"])
            saved = Store(directory).read(result["run_id"] + "-result.json")
            self.assertEqual(saved["status"], "ok")
            self.assertNotIn("جواب بصيغة ترقيم", json.dumps(saved, ensure_ascii=False))

    def test_service_status_survives_worker_and_synthesis_failure_reports(self):
        for failing_stage in ("worker", "synthesis"):
            client = ResearchClient()
            complete = client.complete

            def fail_selected(messages, tools=None):
                text = messages[-1]["content"]
                should_fail = ("المهمة: الذاكرة\n" in text) if failing_stage == "worker" else ("اجمع نتائج" in text)
                if should_fail:
                    raise AppError("تعذر إكمال طلب الخدمة.\nرمز استجابة الخدمة: 502", code="http_502", http_status=502)
                return complete(messages, tools)

            with self.subTest(stage=failing_stage), tempfile.TemporaryDirectory() as directory:
                with patch.object(client, "complete", side_effect=fail_selected), self.assertRaises(AppError) as caught:
                    Agent(client, self.corpus, CFG).research("قارن الذاكرة والعزل والاسترجاع", Store(directory))
                self.assertEqual(caught.exception.http_status, 502)
                path = next(Path(directory).glob("*-result.json"))
                report = read_json(path)
                self.assertEqual(report["http_status"], 502)
                self.assertEqual(report["error_code"], "http_502")
                self.assertEqual(report["stage"], "الفرع 1" if failing_stage == "worker" else "جمع النتائج")
                self.assertEqual(report["research_stats"]["model_calls"], 4 if failing_stage == "worker" else 5)

    def test_one_worker_can_recover_without_sibling_history_or_notes(self):
        client = ResearchClient(fault="worker")
        with tempfile.TemporaryDirectory() as directory:
            result = Agent(client, self.corpus, CFG, notes=["USER_NOTE_SENTINEL"]).research(
                "قارن الذاكرة والعزل والاسترجاع", Store(directory))
            self.assertEqual(len(client.calls), 6)
            self.assertEqual(result["research_stats"]["citation_repairs"], 1)
            workers = [m for m in client.calls if any("المهمة:" in str(x.get("content", "")) for x in m)]
            self.assertTrue(workers)
            self.assertTrue(all("USER_NOTE_SENTINEL" not in str(m) for m in workers))
            initial_workers = [m for m in workers if len(m) == 2]
            self.assertTrue(all("نتيجة فرع مستقل" not in str(m) for m in initial_workers))

    def test_repeated_bad_citations_stop_and_save_safe_diagnostics(self):
        client = ResearchClient(fault="all_workers")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AppError) as caught:
                Agent(client, self.corpus, CFG).research("قارن الذاكرة والعزل والاسترجاع", Store(directory))
            self.assertIn("الفرع", str(caught.exception))
            self.assertNotIn("/بحث", str(caught.exception))
            self.assertEqual(len(client.calls), 5)  # plan + 3 workers + one shared retry; no synthesis
            results = list(Path(directory).glob("research-*-result.json"))
            self.assertEqual(len(results), 1)
            record = read_json(results[0])
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_code"], "unknown_source")
            self.assertEqual(record["research_stats"]["citation_repairs"], 1)
            self.assertEqual(record["research_stats"]["model_calls"], 5)
            saved = "".join(p.read_text(encoding="utf-8") for p in Path(directory).glob("*.json"))
            self.assertNotIn("مسودة مرفوضة لا ينبغي حفظها", saved)
            self.assertNotIn("Snot_retrieved", saved)

    def test_lower_configured_budget_prevents_extra_retry(self):
        client = ResearchClient(fault="all_workers")
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(AppError):
            Agent(client, self.corpus, {**CFG, "max_model_calls": 5}).research(
                "قارن الذاكرة والعزل والاسترجاع", Store(directory))
        self.assertEqual(len(client.calls), 4)  # no reserved room for retry

    def test_too_small_budget_stops_before_request(self):
        client = ResearchClient()
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(AppError):
            Agent(client, self.corpus, {**CFG, "max_model_calls": 3}).research("قارن", Store(directory))
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
