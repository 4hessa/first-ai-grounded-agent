import argparse
import getpass
import json
import os
import platform
import shutil
import sys
import warnings
from pathlib import Path
from threading import Lock
from . import __version__
from .agent import Agent
from .api import NvidiaClient, validate_config
from .knowledge import Corpus
from .storage import AppError, Store, read_json

ROOT = Path(__file__).resolve().parent.parent
PROGRESS_LOCK = Lock()


def show_progress(message):
    # Worker calls may overlap. Each progress line is a host-generated message,
    # never a token stream, provider body, question, key or evidence passage.
    with PROGRESS_LOCK:
        print(message, flush=True)


def show(result):
    print("\n" + result["answer"])
    for warning in result.get("quality_warnings", []):
        print("\nملاحظة على جودة الجواب:", warning)
    if result.get("sources"):
        print("\nفحص المراجع يؤكد مطابقة معرفاتها للمقاطع؛ صحة تفسيرها تحتاج مراجعة النصوص.")
        print("\nالمقاطع التي أشار إليها الجواب:")
        for source in result["sources"]:
            print("[" + source["id"] + "]")
            print(source["file"])
            print(f"الأسطر: {source['start']}–{source['end']}")
    events = result.get("trace", [])
    if events:
        print("\nعدد طلبات النموذج في هذا المسار:", sum(e["event"] == "model" for e in events))
        if result.get("research_stats"):
            print("يشمل العدد التخطيط وجميع الفروع والجمع وأي محاولة تصحيح.")
            print("عدد الفروع:", result["research_stats"]["branches"])
            print("محاولات تصحيح المراجع:", result["research_stats"]["citation_repairs"])
            print("محاولات تصحيح الخطة:", result["research_stats"].get("plan_repairs", 0))
            print("محاولات المراجعة اللغوية:", result["research_stats"].get("language_repairs", 0))
            if "resumed_model_calls" in result["research_stats"]:
                print("منها طلبات الاستكمال الجديدة:", result["research_stats"]["resumed_model_calls"])
                print("أحداث البحث في الملفات أدناه من المحاولة السابقة؛ لم تُنفذ من جديد.")
        else:
            attempts = sum(event["event"] == "language_repair" for event in events)
            if attempts:
                print("يتضمن العدد محاولة مراجعة لغوية واحدة.")
        for event in events:
            if event["event"] == "tool":
                if result.get("research_stats") and event.get("stage"):
                    print("المرحلة:", event["stage"])
                print("بحث نفّذه البرنامج قبل طلب الجواب:" if event.get("origin") == "workflow" else "الأداة:")
                print(event["name"])
                print("نجحت" if event["status"] == "ok" else "رُفضت")
                if "matches" in event:
                    print("عدد المقاطع المسترجعة:", event["matches"])
    if result.get("result_file"):
        print("\nحُفظت النتيجة في:")
        print(result["result_file"])


class DemoClient:
    """A fixed offline fixture, intentionally not presented as an intelligent model."""
    def __init__(self):
        self.step = 0

    def complete(self, messages, tools=None):
        self.step += 1
        if self.step == 1:
            return {"tool_calls": [{"id": "demo_calc_1", "type": "function", "function": {
                "name": "calculate", "arguments": '{"expression":"(12 + 8) / 4"}'}}]}, "tool_calls", {}
        value = json.loads(messages[-1]["content"])["value"]
        return {"content": "النتيجة الحسابية من الأداة: " + str(value)}, "stop", {}


def main():
    parser = argparse.ArgumentParser(description="مساعد تعليمي مبني على أنماط دورة إنفيديا")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="فحص المتطلبات دون اتصال")
    sub.add_parser("demo", help="عرض ثابت للحلقة دون نموذج أو مفتاح")
    for name in ("ask", "grounded", "research"):
        item = sub.add_parser(name)
        item.add_argument("question", nargs="?")
    resume = sub.add_parser("resume", help="استكمال جمع نتائج بحث متوقف من مسوداته المحفوظة")
    resume.add_argument("run_id", help="معرف البحث دون مسار أو امتداد")
    sub.add_parser("chat", help="محادثة في الذاكرة المؤقتة")
    sub.add_parser("index", help="إرسال نصوص المعرفة إلى خدمة التضمين وبناء فهرس")
    notes = sub.add_parser("notes")
    notes.add_argument("action", choices=["list", "add", "remove"])
    notes.add_argument("value", nargs="?")
    args = parser.parse_args()
    cfg = validate_config(read_json(args.config))
    corpus = Corpus(ROOT / "knowledge")
    if args.command == "doctor":
        print("إصدار المساعد:", __version__)
        print("نظام بيئة التشغيل الحالية:")
        print(platform.system())
        print("إصدار بايثون:")
        print(platform.python_version())
        print("عدد مقاطع المعرفة:", len(corpus.chunks))
        print("الإعدادات الأساسية صالحة.")
        print("وجود أدوات النظام لا يثبت أنها تعمل أو أن العزل مفعل:")
        for program in ("nemoclaw", "openshell", "docker"):
            print(program)
            print("موجود" if shutil.which(program) else "غير موجود")
        print("لم يُختبر الاتصال بالنموذج أو العزل في هذا الفحص.")
        return
    if args.command == "demo":
        print("عرض تعليمي ثابت دون شبكة. هذه الاستجابة ليست صادرة عن نموذج ذكاء اصطناعي.")
        show(Agent(DemoClient(), corpus, cfg).run("احسب مجموع اثني عشر وثمانية، ثم اقسمه على أربعة."))
        return
    store = Store(ROOT / "state")
    if args.command == "notes":
        if args.action == "list":
            items = store.notes()
            for number, text in enumerate(items, 1):
                print(f"{number}. {text}")
            if not items:
                print("لا توجد ملاحظات محفوظة.")
        elif args.action == "add":
            print("ستُحفظ الملاحظة محليًا، وتدخل في سياق المحادثات التالية مع مزود النموذج.")
            store.add_note(args.value if args.value is not None else input("الملاحظة: "))
            print("حُفظت الملاحظة.")
        else:
            try:
                number = int(args.value if args.value is not None else input("رقم الملاحظة: "))
            except ValueError:
                raise AppError("أدخل رقم الملاحظة كما يظهر في القائمة.") from None
            store.delete_note(number)
            print("حُذفت الملاحظة.")
        return
    if args.command == "resume":
        saved = Agent(None, corpus, cfg).prepare_resume(args.run_id, store)
        print("سيُستكمل جمع النتائج من المسودات المقبولة، مع إعادة التحقق من مراجعها.")
        print("عدد طلبات المحادثة السابقة:", saved["previous_calls"])
        print("الحد المتبقي لهذه المحاولة، شاملًا التصحيح إن اتسع له:", saved["remaining"])
        print("لا يُعاد التخطيط أو تشغيل الفروع أو طلب التضمين. تُحفظ النتيجة في سجل استكمال منفصل.")
    key = None
    if cfg["provider"] == "nvidia":
        print("وضع الاتصال المباشر للتعلم. سيُرسل السؤال والسياق اللازم إلى خدمة إنفيديا.")
        print("هذا الوضع لا يشغّل عزل إنفيديا تلقائيًا.")
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            if not sys.stdin.isatty():
                raise AppError("شغّل في طرفية تدعم إدخالًا مخفيًا، أو اضبط متغير المفتاح محليًا.")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    key = getpass.getpass("أدخل مفتاح إنفيديا محليًا؛ لن يظهر ولن يُحفظ: ")
            except getpass.GetPassWarning:
                raise AppError("هذه الطرفية لا تدعم إخفاء المفتاح. أوقف الإدخال واستخدم طرفية محلية مناسبة.") from None
    else:
        print("وضع المسار المعزول: يتطلب بيئة قائمة وسياسة تسمح بهذا البرنامج.")
    client = NvidiaClient(cfg, key)
    if args.command == "index":
        print("ستُرسل مقاطع مجلد المعرفة إلى خدمة التضمين. لا تضع أسرارًا ضمنها.")
        index = corpus.build_index(client)
        store.write("index.json", index)
        print("تم بناء الفهرس. عدد المقاطع:", len(index["vectors"]))
        print("لاستخدامه غيّر قيمة إعداد طريقة البحث إلى:")
        print("semantic")
        return
    needs_index = cfg["retrieval"] == "semantic" and args.command != "resume"
    index = store.read("index.json") if needs_index else None
    if needs_index and index is None:
        raise AppError("ابن فهرس التضمين أولًا.")
    agent = Agent(client, corpus, cfg, notes=store.notes(), index=index, progress=show_progress)
    if args.command == "resume":
        show(agent.resume(args.run_id, store))
        return
    if args.command == "chat":
        print("اكتب خروج لإنهاء المحادثة، أو جديد لمسح سياقها. لا تُحفظ المحادثة تلقائيًا.")
        print("قد تُجرى مراجعة لغوية واحدة عند اكتشاف خلط اللغات، وتُحسب ضمن حد الطلبات.")
        print("للبحث في الملفات أولًا اكتب /بحث ثم السؤال في السطر نفسه. يحتفظ السياق بآخر أربعة تبادلات مقبولة.")
        print("قول «احفظ» هنا لا يحفظ ملاحظة. للحفظ استخدم هذا الأمر خارج المحادثة:")
        print("python3 -m first_ai notes add")
        history = []
        while True:
            question = input("\nأنت: ").strip()
            if question == "خروج":
                return
            if question == "جديد":
                history.clear()
                print("بدأت محادثة جديدة. الملاحظات التي حفظتها صراحة ما زالت موجودة.")
                continue
            if not question:
                continue
            try:
                result = agent.run(question, history)
                show(result)
                history.extend([{"role": "user", "content": question},
                                {"role": "assistant", "content": result["answer"]}])
                # Keep complete turns. This is bounded history, not semantic summarization.
                history = history[-8:]
            except AppError as exc:
                print("توقفت هذه المحاولة:", exc)
        return
    question = args.question if args.question is not None else input("السؤال: ")
    if args.command == "research":
        print("سيُحفظ سؤال البحث والمسودات المقبولة ونتيجته أو تقرير التوقف محليًا.")
        print("الحد الأقصى لطلبات المحادثة، شاملًا محاولة تصحيح واحدة للخطة أو المراجع أو اللغة:", min(6, cfg["max_model_calls"]))
        print("قد تضاف ثلاثة طلبات تضمين عند تفعيل البحث الدلالي.")
        result = agent.research(question, store)
    elif args.command == "grounded":
        result = agent.grounded(question)
    else:
        result = agent.run(question)
    show(result)


if __name__ == "__main__":
    try:
        main()
    except (AppError, EOFError, KeyboardInterrupt) as exc:
        print("\n" + (str(exc) if isinstance(exc, AppError) else "توقف التشغيل."), file=sys.stderr)
        sys.exit(1)
