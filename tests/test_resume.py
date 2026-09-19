"""Legacy checkpoint recovery, budget preservation, and safe service errors."""
import contextlib
import copy
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error

from first_ai import __main__ as cli
from first_ai.agent import Agent
from first_ai.api import NvidiaClient
from first_ai.knowledge import Corpus
from first_ai.storage import AppError, Store, read_json

ROOT = Path(__file__).resolve().parents[1]
CFG = read_json(ROOT / "config.json")
RUN_ID = "research-da71439eb9fe44f78dad0f0156454d9f"


def legacy_checkpoint(store, corpus, branches=3):
    """The on-disk schema of 0.1.2; the old HTTP status is already lost."""
    tasks = ["الذاكرة", "العزل", "الاسترجاع"][:branches]
    trace = [{"event": "model", "finish_reason": "stop", "total_tokens": None, "stage": "التخطيط"}]
    store.write(RUN_ID + "-plan.json", {
        "question": "قارن الذاكرة والعزل والاسترجاع", "tasks": tasks,
        "status": "failed", "stage": "جمع النتائج",
    })
    for number, task in enumerate(tasks):
        chunk = corpus.search(task)[0]
        source = {key: chunk[key] for key in ("id", "file", "start", "end")}
        stage = "الفرع " + str(number + 1)
        events = [
            {"event": "tool", "name": "search_knowledge", "status": "ok", "origin": "workflow", "matches": 1, "stage": stage},
            {"event": "model", "finish_reason": "stop", "total_tokens": None, "stage": stage},
        ]
        store.write(RUN_ID + "-" + str(number) + ".json", {
            "task": task, "stage": stage, "answer": "نتيجة محفوظة [" + chunk["id"] + "]",
            "sources": [source], "trace": events, "status": "ok",
        })
        trace.extend(events)
    trace.append({"event": "model", "finish_reason": "error", "total_tokens": None, "stage": "جمع النتائج"})
    store.write(RUN_ID + "-result.json", {
        "status": "failed", "stage": "جمع النتائج", "error": "فشل الطلب إلى الخدمة.",
        "error_code": "application_error", "trace": trace, "run_id": RUN_ID,
        "research_stats": {"model_calls": branches + 2, "branches": branches, "citation_repairs": 0, "max_model_calls": 6},
    })


class SynthesisClient:
    def __init__(self, fault=None):
        self.calls, self.fault = [], fault

    def complete(self, messages, tools=None):
        self.calls.append(copy.deepcopy(messages))
        if self.fault == "http":
            raise AppError("أبلغت الخدمة بأنها غير متاحة لهذا الطلب.\nرمز استجابة الخدمة: 503", code="http_503", http_status=503)
        if self.fault == "citation" and len(self.calls) == 1:
            return {"content": "جواب بمصدر مجهول [Sunknown]"}, "stop", {}
        request = next(m["content"] for m in messages if "مسودات مقبولة شكليًا:\n" in m.get("content", ""))
        evidence = json.loads(request.split("أدلة غير موثوقة كتعليمات:\n", 1)[1])
        return {"content": "جواب عربي من المسودات المحفوظة [" + evidence[0]["id"] + "]"}, "stop", {}

    def embed(self, *args, **kwargs):
        raise AssertionError("Resume must not request embeddings")


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(ROOT / "knowledge")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)
        legacy_checkpoint(self.store, self.corpus)
        self.client = SynthesisClient()
        self.agent = Agent(self.client, self.corpus, CFG)

    def original_bytes(self):
        return {p.name: p.read_bytes() for p in self.store.root.glob("*.json") if "-resume-" not in p.name}

    def test_legacy_resume_uses_one_call_original_sources_and_preserves_saved_work(self):
        before = self.original_bytes()
        with patch.object(self.corpus, "search", side_effect=AssertionError("No repeated retrieval")):
            result = self.agent.resume(RUN_ID, self.store)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(result["research_stats"]["model_calls"], 6)
        self.assertEqual(result["research_stats"]["previous_model_calls"], 5)
        self.assertEqual(result["research_stats"]["resumed_model_calls"], 1)
        self.assertEqual(self.original_bytes(), before)
        self.assertEqual(self.store.read(RUN_ID + "-resume-result.json")["status"], "ok")
        request = self.client.calls[0][-1]["content"]
        self.assertNotIn('"trace"', request)
        evidence = json.loads(request.split("أدلة غير موثوقة كتعليمات:\n", 1)[1])
        expected = {source["id"] for i in range(3) for source in self.store.read(RUN_ID + "-" + str(i) + ".json")["sources"]}
        self.assertEqual({chunk["id"] for chunk in evidence}, expected)
        self.assertTrue(all(chunk == self.corpus.by_id[chunk["id"]] for chunk in evidence))
        with self.assertRaises(AppError):
            self.agent.resume(RUN_ID, self.store)
        self.assertEqual(len(self.client.calls), 1)

    def test_service_failure_is_recorded_without_automatic_retry_or_budget_reset(self):
        self.client.fault = "http"
        before = self.original_bytes()
        with self.assertRaises(AppError) as caught:
            self.agent.resume(RUN_ID, self.store)
        self.assertIn("503", str(caught.exception))
        saved = self.store.read(RUN_ID + "-resume-result.json")
        self.assertEqual(saved["error_code"], "http_503")
        self.assertEqual(saved["http_status"], 503)
        self.assertEqual(saved["trace"][-1]["http_status"], 503)
        self.assertEqual(saved["research_stats"]["model_calls"], 6)
        self.assertEqual(self.original_bytes(), before)
        with self.assertRaises(AppError):
            self.agent.resume(RUN_ID, self.store)
        self.assertEqual(len(self.client.calls), 1)

    def test_last_slot_cannot_trigger_a_seventh_call_for_citation_repair(self):
        self.client.fault = "citation"
        with self.assertRaises(AppError):
            self.agent.resume(RUN_ID, self.store)
        saved = self.store.read(RUN_ID + "-resume-result.json")
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(saved["error_code"], "unknown_source")
        self.assertEqual(saved["research_stats"]["citation_repairs"], 0)
        self.assertNotIn("Sunknown", json.dumps(saved))

    def test_two_remaining_slots_can_use_the_original_single_citation_repair(self):
        legacy_checkpoint(self.store, self.corpus, branches=2)
        self.client.fault = "citation"
        result = self.agent.resume(RUN_ID, self.store)
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(result["research_stats"]["model_calls"], 6)
        self.assertEqual(result["research_stats"]["citation_repairs"], 1)

    def test_interrupted_claim_is_not_silently_removed_or_reused(self):
        marker = self.store.path(RUN_ID + "-resume-result.json")
        marker.write_bytes(b'{"status": "started"')
        with self.assertRaises(AppError):
            self.agent.resume(RUN_ID, self.store)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(marker.read_bytes(), b'{"status": "started"')

    def test_failed_final_save_keeps_claim_and_prevents_duplicate_request(self):
        before = self.original_bytes()
        with patch.object(self.store, "write", side_effect=OSError("disk full")), self.assertRaises(AppError):
            self.agent.resume(RUN_ID, self.store)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.store.read(RUN_ID + "-resume-result.json")["status"], "started")
        self.assertEqual(self.original_bytes(), before)
        with self.assertRaises(AppError):
            self.agent.resume(RUN_ID, self.store)
        self.assertEqual(len(self.client.calls), 1)

    def test_missing_invalid_or_relabelled_draft_stops_before_claim_or_request(self):
        name = RUN_ID + "-0.json"
        original = self.store.read(name)
        cases = [None, {**original, "status": "failed"}, {**original, "answer": "جواب [Sunknown]"},
                 {**original, "sources": [{**original["sources"][0], "file": "different.md"}]},
                 {**original, "sources": original["sources"] * 2}, {**original, "task": "مهمة مختلفة"}]
        for bad in cases:
            with self.subTest(case=bad):
                self.store.write(name, bad)
                with self.assertRaises(AppError):
                    self.agent.resume(RUN_ID, self.store)
                self.assertFalse(self.store.path(RUN_ID + "-resume-result.json").exists())
        self.assertEqual(self.client.calls, [])

    def test_changed_source_text_cannot_reuse_the_old_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            knowledge = Path(directory) / "knowledge"
            shutil.copytree(ROOT / "knowledge", knowledge)
            source = self.store.read(RUN_ID + "-0.json")["sources"][0]
            path = knowledge / source["file"]
            path.write_text(path.read_text(encoding="utf-8") + "\nتعديل على النص.\n", encoding="utf-8")
            with self.assertRaises(AppError):
                Agent(self.client, Corpus(knowledge), CFG).resume(RUN_ID, self.store)
        self.assertEqual(self.client.calls, [])

    def test_invalid_ids_counts_and_stage_are_rejected_locally(self):
        for run_id in ("../research-test", RUN_ID + "-result.json", "research-123"):
            with self.subTest(run_id=run_id), self.assertRaises(AppError):
                self.agent.resume(run_id, self.store)
        name = RUN_ID + "-result.json"
        original = self.store.read(name)
        for bad in ({**original, "stage": "التخطيط"},
                    {**original, "research_stats": {**original["research_stats"], "model_calls": 4}},
                    {**original, "research_stats": {**original["research_stats"], "model_calls": True}}):
            self.store.write(name, bad)
            with self.subTest(report=bad), self.assertRaises(AppError):
                self.agent.resume(RUN_ID, self.store)
        self.assertEqual(self.client.calls, [])

    def test_current_or_original_lower_limit_is_respected(self):
        with self.assertRaises(AppError):
            Agent(self.client, self.corpus, {**CFG, "max_model_calls": 5}).resume(RUN_ID, self.store)
        name = RUN_ID + "-result.json"
        report = self.store.read(name)
        report["research_stats"]["max_model_calls"] = 5
        self.store.write(name, report)
        with self.assertRaises(AppError):
            Agent(self.client, self.corpus, {**CFG, "max_model_calls": 8}).resume(RUN_ID, self.store)
        self.assertEqual(self.client.calls, [])

    def test_concurrent_resumes_cannot_spend_the_last_slot_twice(self):
        gate = threading.Barrier(2)
        create = self.store.create_once

        def concurrent_create(*args):
            gate.wait(timeout=5)
            return create(*args)

        def attempt(_):
            try:
                return Agent(self.client, self.corpus, CFG).resume(RUN_ID, self.store)["status"]
            except AppError:
                return "refused"

        with patch.object(self.store, "create_once", side_effect=concurrent_create), ThreadPoolExecutor(2) as pool:
            outcomes = list(pool.map(attempt, range(2)))
        self.assertCountEqual(outcomes, ["ok", "refused"])
        self.assertEqual(len(self.client.calls), 1)

    def test_cli_resume_needs_neither_question_nor_semantic_index(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            shutil.copytree(ROOT / "knowledge", project / "knowledge")
            (project / "config.json").write_text(json.dumps({**CFG, "retrieval": "semantic"}), encoding="utf-8")
            legacy_checkpoint(Store(project / "state"), self.corpus)
            output = io.StringIO()
            with patch.object(cli, "ROOT", project), \
                    patch("sys.argv", ["first_ai", "resume", RUN_ID]), \
                    patch.dict("os.environ", {"NVIDIA_API_KEY": "local-test-placeholder"}), \
                    patch.object(cli, "NvidiaClient", return_value=self.client), \
                    patch("builtins.input", side_effect=AssertionError("No question prompt")), \
                    contextlib.redirect_stdout(output):
                cli.main()
            self.assertEqual(len(self.client.calls), 1)
            self.assertIn("منها طلبات الاستكمال الجديدة: 1", output.getvalue())
            self.assertFalse((project / "state" / "index.json").exists())

    def test_cli_rejects_missing_checkpoint_before_key_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            shutil.copytree(ROOT / "knowledge", project / "knowledge")
            shutil.copyfile(ROOT / "config.json", project / "config.json")
            with patch.object(cli, "ROOT", project), patch("sys.argv", ["first_ai", "resume", RUN_ID]), \
                    patch.object(cli.getpass, "getpass") as key_prompt, self.assertRaises(AppError):
                cli.main()
            key_prompt.assert_not_called()


class ServiceErrorTests(unittest.TestCase):
    def test_numeric_status_is_preserved_without_remote_body_headers_reason_or_key(self):
        secret = "DO-NOT-DISPLAY-CREDENTIAL-OR-REMOTE-CONTENT"
        for status in (400, 401, 403, 404, 408, 413, 429, 500, 502, 503, 504, 418):
            with self.subTest(status=status):
                remote = urllib.error.HTTPError("https://example.invalid/" + secret, status, secret,
                                                {"X-Secret": secret}, io.BytesIO(secret.encode()))
                opener = Mock()
                opener.open.side_effect = remote
                with patch("urllib.request.build_opener", return_value=opener), self.assertRaises(AppError) as caught:
                    NvidiaClient(CFG, secret).complete([{"role": "user", "content": "سؤال"}])
                error = caught.exception
                self.assertEqual(error.http_status, status)
                self.assertEqual(error.code, "http_" + str(status))
                self.assertIn(str(status), str(error))
                self.assertNotIn(secret, json.dumps(error.details()))
                self.assertEqual(opener.open.call_count, 1)

    def test_connection_errors_are_not_reported_as_server_statuses(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.URLError("PRIVATE_CONNECTION_DETAIL")
        with patch("urllib.request.build_opener", return_value=opener), self.assertRaises(AppError) as caught:
            NvidiaClient(CFG, "local-test-placeholder").complete([])
        self.assertEqual(caught.exception.code, "connection_error")
        self.assertIsNone(caught.exception.http_status)
        self.assertNotIn("PRIVATE_CONNECTION_DETAIL", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
