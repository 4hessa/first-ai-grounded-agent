"""Behavior tests for bounded planner/language recovery and observable progress."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from first_ai.__main__ import show, show_progress
from first_ai.agent import Agent
from first_ai.knowledge import Corpus
from first_ai.planning import PlanError, read_plan
from first_ai.storage import AppError, Store, read_json
from test_agent import Scripted, tool
from test_research import ResearchClient

ROOT = Path(__file__).resolve().parents[1]
CFG = read_json(ROOT / "config.json")


def reply(text):
    return {"content": text}, "stop", {}


class PlannerValidationTests(unittest.TestCase):
    def test_raw_or_one_outer_fence_preserves_task_text(self):
        tasks = ["شرح عزل التشغيل", "شرح عزل سياق الوكلاء"]
        body = json.dumps({"tasks": tasks}, ensure_ascii=False)
        for text in (body, " \n" + body + "\n", "```json\n" + body + "\n```", "```\n" + body + "\n```"):
            with self.subTest(text=text):
                self.assertEqual(read_plan({"content": text}, "stop"), tasks)

    def test_invalid_plans_have_specific_reasons(self):
        cases = [
            ("", "plan_empty"), (None, "plan_empty"),
            ("x" * 12001, "plan_too_long"),
            ('مقدمة {"tasks":["أ","ب"]}', "plan_json"),
            ('{"tasks":["أ","ب"]} trailing', "plan_json"),
            ('{"tasks":["أ","ب"],"tasks":["ج","د"]}', "plan_json"),
            ('{"tasks": NaN}', "plan_json"),
            ('{"tasks":["أ","ب"],"shell":"bad"}', "plan_schema"),
            ('["أ","ب"]', "plan_schema"),
            ('{"tasks":"أ"}', "plan_schema"),
            ('{"tasks":["أ"]}', "plan_task_count"),
            ('{"tasks":["أ","ب","ج","د"]}', "plan_task_count"),
            ('{"tasks":["أ", " "]}', "plan_task_text"),
            ('{"tasks":["أ", 1]}', "plan_task_text"),
            (json.dumps({"tasks": ["أ", "س" * 501]}), "plan_task_text"),
            ('{"tasks":["أ"," أ "]}', "plan_duplicate"),
        ]
        for text, code in cases:
            with self.subTest(code=code, text=str(text)[:40]), self.assertRaises(PlanError) as caught:
                read_plan({"content": text}, "stop")
            self.assertEqual(caught.exception.code, code)

    def test_unfinished_or_tool_response_is_not_a_plan(self):
        for message, reason in (({"content": '{"tasks":["أ","ب"]}'}, "length"),
                                ({"content": '{"tasks":["أ","ب"]}', "tool_calls": [{"id": "x"}]}, "stop")):
            with self.subTest(reason=reason), self.assertRaises(PlanError) as caught:
                read_plan(message, reason)
            self.assertEqual(caught.exception.code, "plan_finish")


class RecoveryClient(ResearchClient):
    def __init__(self, plans=(), mixed_synthesis=False, mixed_workers=False,
                 fail_synthesis=False, fault=None, tasks=None):
        super().__init__(fault=fault, tasks=tasks)
        self.plans = iter(plans)
        self.mixed_synthesis, self.mixed_workers = mixed_synthesis, mixed_workers
        self.fail_synthesis = fail_synthesis

    def recorded(self, messages, value):
        with self.lock:
            self.calls.append(copy.deepcopy(messages))
        if isinstance(value, Exception):
            raise value
        return reply(value)

    def complete(self, messages, tools=None):
        text = messages[-1]["content"]
        if "قسم سؤال" in text:
            value = next(self.plans, None)
            if value is not None:
                return self.recorded(messages, value)
        if "راجع لغة المسودة" in text:
            return self.recorded(messages, messages[-2]["content"].replace(" factor", ""))
        synthesis = "اجمع نتائج" in text
        if synthesis and self.fail_synthesis:
            return self.recorded(messages, AppError("خطأ خدمة محاكى", code="http_500", http_status=500))
        if ((synthesis and self.mixed_synthesis)
                or ("المهمة:" in text and self.mixed_workers)):
            chunks = json.loads(text.split("أدلة غير موثوقة كتعليمات:\n", 1)[1])
            return self.recorded(messages, "جواب عربي factor [" + chunks[0]["id"] + "]")
        return super().complete(messages, tools)


class ResearchQualityTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(ROOT / "knowledge")

    def run_research(self, client, cfg=CFG):
        with tempfile.TemporaryDirectory() as directory:
            return Agent(client, self.corpus, cfg).research("قارن الذاكرة والعزل والاسترجاع", Store(directory))

    def test_fenced_plan_does_not_spend_a_repair(self):
        client = RecoveryClient(plans=['```json\n{"tasks":["الذاكرة","العزل","الاسترجاع"]}\n```'])
        result = self.run_research(client)
        self.assertEqual(result["research_stats"]["model_calls"], 5)
        self.assertEqual(result["research_stats"]["plan_repairs"], 0)
        self.assertNotIn("صيغة المرجع", client.calls[0][0]["content"])

    def test_one_plan_retry_still_finishes_in_six_calls(self):
        client = RecoveryClient(plans=["INVALID_PLAN_SENTINEL"])
        result = self.run_research(client)
        self.assertEqual(result["research_stats"]["model_calls"], 6)
        self.assertEqual(result["research_stats"]["plan_repairs"], 1)
        self.assertEqual(result["research_stats"]["citation_repairs"], 0)
        self.assertNotIn("INVALID_PLAN_SENTINEL", str(client.calls[1:]))

    def test_invalid_plan_twice_stops_before_any_worker(self):
        client = RecoveryClient(plans=["PRIVATE_PLAN_SENTINEL", "PRIVATE_PLAN_SENTINEL"])
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(AppError) as caught:
            store = Store(directory)
            try:
                Agent(client, self.corpus, CFG).research("قارن", store)
            finally:
                saved = read_json(next(Path(directory).glob("*-result.json")))
                all_saved = "".join(p.read_text() for p in Path(directory).glob("*.json"))
                self.assertEqual(saved["stage"], "التخطيط")
                self.assertEqual(saved["error_code"], "plan_json")
                self.assertEqual(saved["research_stats"]["model_calls"], 2)
                self.assertEqual(saved["research_stats"]["branches"], 0)
                self.assertNotIn("PRIVATE_PLAN_SENTINEL", all_saved)
        self.assertEqual(caught.exception.code, "plan_json")
        self.assertEqual(len(client.calls), 2)

    def test_four_call_budget_allows_two_tasks_but_no_plan_retry(self):
        cfg = {**CFG, "max_model_calls": 4}
        good = RecoveryClient(tasks=["الذاكرة", "العزل"])
        self.assertEqual(self.run_research(good, cfg)["research_stats"]["model_calls"], 4)
        self.assertIn("بحد أقصى 2", good.calls[0][-1]["content"])
        bad = RecoveryClient(plans=["bad"])
        with self.assertRaises(AppError):
            self.run_research(bad, cfg)
        self.assertEqual(len(bad.calls), 1)

    def test_five_call_budget_retries_plan_with_only_two_tasks(self):
        client = RecoveryClient(plans=["bad"], tasks=["الذاكرة", "العزل"])
        result = self.run_research(client, {**CFG, "max_model_calls": 5})
        self.assertEqual(result["research_stats"]["model_calls"], 5)
        self.assertIn("بحد أقصى 2", client.calls[1][-1]["content"])

    def test_plan_retry_does_not_allow_an_extra_citation_retry(self):
        client = RecoveryClient(plans=["bad"], fault="synthesis")
        with self.assertRaises(AppError):
            self.run_research(client)
        self.assertEqual(len(client.calls), 6)
        self.assertFalse(any("أعد صياغة الجواب" in messages[-1]["content"] for messages in client.calls))

    def test_worker_language_is_checked_but_only_final_language_is_retried(self):
        client = RecoveryClient(mixed_workers=True, mixed_synthesis=True)
        result = self.run_research(client)
        self.assertEqual(result["research_stats"]["model_calls"], 6)
        self.assertEqual(result["research_stats"]["language_repairs"], 1)
        self.assertFalse(result["quality_warnings"])
        self.assertEqual(sum("راجع لغة المسودة" in messages[-1]["content"] for messages in client.calls), 1)
        self.assertIn("quality_warnings", str(client.calls[-2]))

    def test_plan_and_final_language_share_one_correction(self):
        client = RecoveryClient(plans=["bad"], mixed_synthesis=True)
        result = self.run_research(client)
        self.assertEqual(result["research_stats"]["model_calls"], 6)
        self.assertEqual(result["research_stats"]["plan_repairs"], 1)
        self.assertEqual(result["research_stats"]["language_repairs"], 0)
        self.assertTrue(result["quality_warnings"])

    def test_service_failure_does_not_retry_as_bad_plan(self):
        client = RecoveryClient(plans=[AppError("خطأ خدمة محاكى", code="http_500", http_status=500)])
        with self.assertRaises(AppError) as caught:
            self.run_research(client)
        self.assertEqual(caught.exception.http_status, 500)
        self.assertEqual(len(client.calls), 1)

    def test_resume_preserves_used_plan_correction_and_original_files(self):
        client = RecoveryClient(plans=["bad"], tasks=["الذاكرة", "العزل"], fail_synthesis=True)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            with self.assertRaises(AppError):
                Agent(client, self.corpus, CFG).research("قارن الذاكرة والعزل", store)
            report = read_json(next(Path(directory).glob("*-result.json")))
            self.assertEqual(report["research_stats"]["model_calls"], 5)
            before = {p: p.read_bytes() for p in Path(directory).glob("*.json")}
            resumed_client = RecoveryClient(mixed_synthesis=True)
            result = Agent(resumed_client, self.corpus, CFG).resume(report["run_id"], store)
            self.assertEqual(result["research_stats"]["model_calls"], 6)
            self.assertEqual(result["research_stats"]["plan_repairs"], 1)
            self.assertEqual(result["research_stats"]["language_repairs"], 0)
            self.assertEqual(len(resumed_client.calls), 1)
            self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_resume_refuses_mismatched_or_multiple_correction_counters(self):
        from test_resume import legacy_checkpoint, RUN_ID
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            legacy_checkpoint(store, self.corpus)
            original = store.read(RUN_ID + "-result.json")
            for values in ({"plan_repairs": 1}, {"language_repairs": True},
                           {"plan_repairs": 1, "language_repairs": 1}):
                modified = copy.deepcopy(original)
                modified["research_stats"].update(values)
                store.write(RUN_ID + "-result.json", modified)
                client = RecoveryClient()
                with self.subTest(values=values), self.assertRaises(AppError):
                    Agent(client, self.corpus, CFG).resume(RUN_ID, store)
                self.assertFalse(client.calls)
                self.assertFalse(store.path(RUN_ID + "-resume-result.json").exists())


class LanguageRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus(ROOT / "knowledge")

    def test_mixed_chat_gets_one_rewrite_and_clean_chat_does_not(self):
        for first, expected_calls in (("هذا شرح factor", 2), ("هذا شرح بالعربية", 1)):
            client = Scripted([reply(first), reply("هذا شرح بالعربية")])
            result = Agent(client, self.corpus, CFG).run("اشرح")
            self.assertEqual(len(client.messages), expected_calls)
            self.assertFalse(result["quality_warnings"])

    def test_same_required_latin_term_is_line_separated_after_rewrite(self):
        original = "واجهة برمجة التطبيقات (API) تتيح التواصل."
        client = Scripted([reply(original), reply(original)])
        result = Agent(client, self.corpus, CFG).run("اشرح واجهة برمجة التطبيقات")
        self.assertEqual(len(client.messages), 2)
        self.assertFalse(result["quality_warnings"])
        self.assertIn("\n(API)\n", result["answer"])
        self.assertEqual(result["trace"][-1]["status"], "accepted")

    def test_reviewer_notes_wrapper_is_not_shown_to_user(self):
        original = "واجهة برمجة التطبيقات (API) تتيح التواصل."
        revised = ("ملاحظات المسودة السابقة:\n"
                   "- احتفظ بالمصطلح الضروري.\n\n"
                   "الجواب بعد المراجعة:\n"
                   "واجهة برمجة التطبيقات\nAPI\nتتيح التواصل.")
        client = Scripted([reply(original), reply(revised)])
        result = Agent(client, self.corpus, CFG).run("اشرح واجهة برمجة التطبيقات")
        self.assertEqual(len(client.messages), 2)
        self.assertFalse(result["quality_warnings"])
        self.assertEqual(result["answer"], "واجهة برمجة التطبيقات\nAPI\nتتيح التواصل.")
        self.assertNotIn("ملاحظات المسودة السابقة", result["answer"])
        self.assertNotIn("الجواب بعد المراجعة", result["answer"])
        self.assertEqual(result["trace"][-1]["status"], "accepted")

    def test_acronym_layout_falls_back_to_original_when_reviewer_stays_mixed(self):
        original = "واجهة برمجة التطبيقات (API) تتيح التواصل."
        revised = "واجهة برمجة التطبيقات API interface تتيح التواصل."
        client = Scripted([reply(original), reply(revised)])
        result = Agent(client, self.corpus, CFG).run("اشرح واجهة برمجة التطبيقات")
        self.assertEqual(len(client.messages), 2)
        self.assertFalse(result["quality_warnings"])
        self.assertIn("\n(API)\n", result["answer"])
        self.assertNotIn("interface", result["answer"])
        self.assertEqual(result["trace"][-1]["status"], "accepted")
        self.assertEqual(result["trace"][-1]["mode"], "local_layout_fallback")

    def test_alternate_review_answer_header_is_removed(self):
        original = "واجهة برمجة التطبيقات (API) تتيح التواصل."
        revised = "مراجعة مختصرة.\nالجواب النهائي:\nواجهة برمجة التطبيقات\nAPI\nتتيح التواصل."
        client = Scripted([reply(original), reply(revised)])
        result = Agent(client, self.corpus, CFG).run("اشرح واجهة برمجة التطبيقات")
        self.assertFalse(result["quality_warnings"])
        self.assertEqual(result["answer"], "واجهة برمجة التطبيقات\nAPI\nتتيح التواصل.")

    def test_duplicate_acronym_line_is_removed_after_local_fallback(self):
        original = "واجهة برمجة التطبيقات (API) تسمح للتطبيقات.\nAPI"
        revised = "واجهة برمجة التطبيقات API interface تسمح للتطبيقات."
        client = Scripted([reply(original), reply(revised)])
        result = Agent(client, self.corpus, CFG).run("اشرح واجهة برمجة التطبيقات")
        self.assertEqual(len(client.messages), 2)
        self.assertFalse(result["quality_warnings"])
        self.assertEqual(result["answer"], "واجهة برمجة التطبيقات\n(API)\nتسمح للتطبيقات.")
        self.assertEqual(result["trace"][-1]["mode"], "local_layout_fallback")

    def test_duplicate_acronym_line_is_removed_from_clean_reviewer_answer(self):
        original = "واجهة برمجة التطبيقات (API) تسمح للتطبيقات."
        revised = "واجهة برمجة التطبيقات\n(API)\nتسمح للتطبيقات.\nAPI"
        client = Scripted([reply(original), reply(revised)])
        result = Agent(client, self.corpus, CFG).run("اشرح واجهة برمجة التطبيقات")
        self.assertEqual(len(client.messages), 2)
        self.assertFalse(result["quality_warnings"])
        self.assertEqual(result["answer"], "واجهة برمجة التطبيقات\n(API)\nتسمح للتطبيقات.")

    def test_failed_or_still_mixed_rewrite_keeps_original_and_warning(self):
        original = "هذا شرح factor"
        for candidate in (reply("هذا شرح altro"), reply("A short answer."), reply("رد [Sunknown]"),
                          ({"content": "رد غير مكتمل"}, "length", {}),
                          tool("calculate", {"expression": "2+2"})):
            with self.subTest(candidate=candidate):
                client = Scripted([reply(original), candidate])
                result = Agent(client, self.corpus, CFG).run("اشرح")
                self.assertEqual(result["answer"], original)
                self.assertTrue(result["quality_warnings"])
                self.assertEqual(len(client.messages), 2)
                self.assertFalse(any(event["event"] == "tool" for event in result["trace"]))

    def test_language_rewrite_cannot_change_numbers_code_or_urls(self):
        cases = [
            ("القيمة factor 5", "القيمة 6"),
            ("المثال factor `x = 5`", "المثال `x = 6`"),
            ("رابط factor https://example.com/a", "رابط https://example.com/b"),
        ]
        for original, changed in cases:
            with self.subTest(original=original):
                client = Scripted([reply(original), reply(changed)])
                result = Agent(client, self.corpus, CFG).run("اشرح")
                self.assertEqual(result["answer"], original)
                self.assertEqual(result["trace"][-1]["reason"], "language_changed_literals")

    def test_grounded_rewrite_keeps_exact_cited_sources(self):
        chunks = self.corpus.search("العزل")
        first, other = [chunk["id"] for chunk in chunks[:2]]
        original = "شرح عربي factor [" + first + "]"
        for source, accepted in ((first, True), (other, False), ("Sunknown", False)):
            with self.subTest(source=source):
                client = Scripted([reply(original), reply("شرح عربي [" + source + "]")])
                result = Agent(client, self.corpus, CFG).grounded("العزل")
                self.assertEqual(result["sources"][0]["id"], first)
                self.assertEqual(bool(result["quality_warnings"]), not accepted)
                self.assertEqual(len(client.messages), 2)

    def test_spent_call_budget_prevents_language_request(self):
        client = Scripted([reply("هذا شرح factor")])
        result = Agent(client, self.corpus, {**CFG, "max_model_calls": 1}).run("اشرح")
        self.assertEqual(len(client.messages), 1)
        self.assertTrue(result["quality_warnings"])

    def test_language_request_after_tools_cannot_exceed_total_budget(self):
        client = Scripted([tool("calculate", {"expression": "2+3"}), reply("القيمة factor 5")])
        result = Agent(client, self.corpus, {**CFG, "max_model_calls": 2}).run("احسب")
        self.assertEqual(len(client.messages), 2)
        self.assertEqual(sum(event["event"] == "model" for event in result["trace"]), 2)
        self.assertTrue(result["quality_warnings"])

    def test_language_http_error_counts_once_without_repeating_original_request(self):
        class FailRewrite(Scripted):
            def complete(self, messages, tools=None):
                if self.messages:
                    self.messages.append(copy.deepcopy(messages))
                    raise AppError("خطأ خدمة محاكى", code="http_500", http_status=500)
                return super().complete(messages, tools)
        client = FailRewrite([reply("هذا شرح factor")])
        result = Agent(client, self.corpus, CFG).run("اشرح")
        self.assertEqual(len(client.messages), 2)
        self.assertEqual(result["answer"], "هذا شرح factor")
        self.assertEqual(result["trace"][-2]["http_status"], 500)

    def test_large_rewrite_context_does_not_spend_a_request(self):
        client = Scripted([reply("هذا شرح factor " + "ع" * 10000)])
        result = Agent(client, self.corpus, {**CFG, "max_context_chars": 6000}).run("اشرح")
        self.assertEqual(len(client.messages), 1)
        self.assertTrue(result["quality_warnings"])

    def test_progress_is_visible_before_request_and_never_prints_content(self):
        progress = []
        class Observe(Scripted):
            def complete(self, messages, tools=None):
                self_before = len(progress)
                assert self_before >= 1
                assert "جاري تنفيذ" in progress[-1]
                return super().complete(messages, tools)
        client = Observe([reply("جواب")])
        Agent(client, self.corpus, CFG, notes=["PRIVATE_NOTE"], progress=progress.append).run("PRIVATE_QUESTION")
        self.assertEqual(len(progress), 2)
        self.assertNotIn("PRIVATE", str(progress))
        self.assertNotIn("جواب", str(progress))

    def test_cli_displays_language_attempt_without_hiding_total(self):
        client = Scripted([reply("شرح factor"), reply("شرح بالعربية")])
        result = Agent(client, self.corpus, CFG).run("اشرح")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            show_progress("رسالة تقدم")
            show(result)
        self.assertIn("عدد طلبات النموذج في هذا المسار: 2", output.getvalue())
        self.assertIn("محاولة مراجعة لغوية واحدة", output.getvalue())


if __name__ == "__main__":
    unittest.main()
