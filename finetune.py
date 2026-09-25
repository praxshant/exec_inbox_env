# finetune.py — turn the env's ground truth into an OpenAI fine-tuning dataset.
#
# The env already knows the correct label/priority for every email
# (env/tasks.py), so we get supervised data for free, in the exact single-call
# format llm_agent uses at inference. A model fine-tuned on this needs no
# few-shot examples in the prompt -> shorter prompts -> lower latency and cost.
#   Build:   python finetune.py             (writes finetune.jsonl, no API key)
#   Launch:  python finetune.py --launch     (needs OPENAI_API_KEY)

import json
import sys

from llm_agent import SYSTEM_PROMPT, _compact
from inference import generate_reply
from env.tasks import TASKS

OUT = "finetune.jsonl"


def _target(email):
    """Correct plan entry, straight from ground truth."""
    reply = generate_reply(email.subject, email.body) if email.label == "important" else ""
    return {"id": email.id, "label": email.label,
            "priority": email.priority or "low", "reply": reply}


def _example(public_inbox, plan_items):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(_compact(public_inbox))},
        {"role": "assistant", "content": json.dumps({"plan": plan_items})},
    ]}


def build_dataset(path=OUT):
    rows = []
    for emails in TASKS.values():
        pub = [{"id": e.id, "sender": e.sender, "subject": e.subject, "body": e.body}
               for e in emails]
        # One whole-inbox example (matches real inference) + one per email for
        # volume. A real fine-tune wants many more real inboxes here.
        rows.append(_example(pub, [_target(e) for e in emails]))
        for e, pe in zip(emails, pub):
            rows.append(_example([pe], [_target(e)]))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def create_job(path=OUT, base_model="gpt-4o-mini-2024-07-18"):
    """Upload the dataset and start a fine-tune job. Needs OPENAI_API_KEY."""
    from openai import OpenAI
    client = OpenAI()
    up = client.files.create(file=open(path, "rb"), purpose="fine-tune")
    job = client.fine_tuning.jobs.create(training_file=up.id, model=base_model)
    print(f"fine-tune job started: {job.id} (base={base_model})")
    print("poll: client.fine_tuning.jobs.retrieve(<id>).status; then set MODEL_NAME to the result")
    return job.id


if __name__ == "__main__":
    if "--launch" in sys.argv:
        create_job()
    else:
        n = build_dataset()
        print(f"wrote {n} examples -> {OUT}")
        with open(OUT, encoding="utf-8") as f:
            print("first example:", f.readline().strip()[:220], "...")
