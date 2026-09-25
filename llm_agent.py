# llm_agent.py — LLM triage policy + a runnable proof of the optimization.
#
# The win: ONE structured LLM call per EPISODE (whole inbox -> full plan)
# instead of the old one-hint-call-PER-STEP. Fewer calls = lower latency and
# cost. With no LLM configured it falls back to heuristics at zero latency.
#   Proof:  python llm_agent.py

import json
import time
from types import SimpleNamespace

SYSTEM_PROMPT = (
    "You are an executive-inbox triage engine. For EVERY email decide: "
    "label (important|spam|promo), priority (low|medium|high), and reply "
    "(one short line if it is important and worth a reply, else empty). "
    "Respond with COMPACT JSON only, no prose: "
    '{"plan":[{"id":"e1","label":"important","priority":"high","reply":"..."}]}'
)

PRIO_RANK = {"high": 0, "medium": 1, "low": 2}


def _compact(inbox):
    # Send only the fields that matter, truncate bodies -> fewer input tokens.
    return [
        {"id": e["id"], "sender": e["sender"],
         "subject": e["subject"], "body": e["body"][:200]}
        for e in inbox
    ]


def _heuristic_plan(inbox):
    # Zero-latency fallback / mock brain. Lazy import avoids an import cycle.
    from inference import classify_email, prioritize_email, generate_reply
    plan = {}
    for e in inbox:
        label = classify_email(e["sender"], e["subject"], e["body"])
        prio = prioritize_email(e["sender"], e["subject"], e["body"], label)
        reply = generate_reply(e["subject"], e["body"]) if label == "important" else ""
        plan[e["id"]] = {"label": label, "priority": prio, "reply": reply}
    return plan

def _llm_plan(inbox, llm, model):
    resp = llm.chat.completions.create(
        model=model,
        temperature=0,                             # deterministic -> no retries
        max_tokens=600,
        response_format={"type": "json_object"},   # structured -> no parse retries
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(_compact(inbox))},
        ],
    )
    try:
        raw = json.loads(resp.choices[0].message.content)
        return {r["id"]: {"label": r["label"], "priority": r["priority"],
                          "reply": r.get("reply", "")} for r in raw["plan"]}
    except Exception:
        return _heuristic_plan(inbox)              # never crash the episode


_PLAN_CACHE = {}


def plan_inbox(observation, llm=None, model="gpt-4o-mini"):
    """ONE call: whole inbox -> {id: {label, priority, reply}}.
    Cached by inbox signature so identical/repeat inboxes cost zero calls."""
    inbox = observation.get("inbox", [])
    if not inbox:
        return {}
    key = tuple(sorted(e["id"] for e in inbox))
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]
    plan = _llm_plan(inbox, llm, model) if llm is not None else _heuristic_plan(inbox)
    _PLAN_CACHE[key] = plan
    return plan

def plan_action(observation, plan, replied, classified, prioritized, handled):
    """Emit ONE action per step from the precomputed plan. No LLM here."""
    inbox = observation.get("inbox", [])

    def rank(e):
        p = plan.get(e["id"], {})
        imp = 0 if p.get("label") == "important" else 1
        return (imp, PRIO_RANK.get(p.get("priority"), 3))

    for e in sorted(inbox, key=rank):
        eid = e["id"]
        p = plan.get(eid) or _heuristic_plan([e])[eid]
        label, prio, reply = p["label"], p["priority"], p.get("reply", "")
        # Reply to high-priority important first — beat the deadlines.
        if label == "important" and prio == "high" and eid not in replied:
            replied.add(eid); handled.add(eid)
            return {"type": "reply", "email_id": eid,
                    "content": reply or "Confirmed. I will handle this promptly. Thank you."}
        if eid not in classified:
            classified.add(eid)
            return {"type": "classify", "email_id": eid, "label": label}
        if eid not in prioritized:
            prioritized.add(eid)
            return {"type": "prioritize", "email_id": eid, "priority": prio}
        if eid not in handled:
            handled.add(eid)
            if label in ("spam", "promo"):
                return {"type": "archive", "email_id": eid}
            return {"type": "reply", "email_id": eid,
                    "content": reply or "Confirmed. I will handle this promptly. Thank you."}
    return {"type": "noop"}

class MockLLM:
    """Offline stand-in for an OpenAI client. Simulates a fixed network
    round-trip and answers with the heuristic, so the benchmark measures the
    thing we actually optimize — CALL COUNT and LATENCY — not model IQ."""

    def __init__(self, latency_s=0.15):
        self.latency_s = latency_s
        self.calls = 0
        self.in_tokens = 0
        self.chat = self          # so llm.chat.completions.create(...) resolves
        self.completions = self

    def create(self, model, messages, **kw):
        self.calls += 1
        text = "".join(m["content"] for m in messages)
        self.in_tokens += len(text) // 4          # ~4 chars/token
        time.sleep(self.latency_s)                # simulate a round-trip
        try:
            plan = _heuristic_plan(json.loads(messages[-1]["content"]))
            content = json.dumps({"plan": [{"id": k, **v} for k, v in plan.items()]})
        except Exception:
            content = "urgent: reply now"         # per-step hint mode
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _score(env, task):
    from env.grader import grade_episode
    from env.tasks import TASKS
    return grade_episode(env.get_action_history(), TASKS[task])["final_score"]


def _run(task, llm, per_step):
    """Same policy either way. per_step rebuilds the plan EVERY step (the old
    waste); batched builds it once. Identical actions -> identical score, so
    the only thing that moves is calls / tokens / latency."""
    from env.environment import EmailEnv
    from env.models import Action
    env = EmailEnv(task); obs = env.reset().model_dump()
    plan = None if per_step else plan_inbox(obs, llm, "mock")   # ONE call
    r, c, p, h = set(), set(), set(), set()
    for _ in range(env.max_steps):
        if per_step:                          # naive: re-call the LLM each step
            _PLAN_CACHE.clear()
            plan = plan_inbox(obs, llm, "mock")
        act = plan_action(obs, plan, r, c, p, h)
        o, _, done, _ = env.step(Action(**act)); obs = o.model_dump()
        if done:
            break
    return _score(env, task)


def benchmark():
    hdr = f"{'strategy':<22}{'llm_calls':>10}{'in_tokens':>11}{'wall_s':>9}{'score':>8}"
    print(hdr); print("-" * len(hdr))
    for task in ("easy", "medium", "hard"):
        for name, per_step in (("per_step_llm", True), ("batched_llm", False)):
            _PLAN_CACHE.clear()
            llm = MockLLM()
            t0 = time.perf_counter()
            score = _run(task, llm, per_step)
            dt = time.perf_counter() - t0
            label = f"{task}/{name}"
            print(f"{label:<22}{llm.calls:>10}{llm.in_tokens:>11}{dt:>9.2f}{score:>8.3f}")



if __name__ == "__main__":
    benchmark()



