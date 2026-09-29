---
title: Exec Inbox Env
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ExecInbox — AI Executive Assistant Environment

An OpenEnv-compatible inbox management simulation where an AI agent acts as an executive assistant.

## Overview

The agent must:
- **Classify** emails as `important`, `spam`, or `promo`
- **Prioritize** emails as `high`, `medium`, or `low`
- **Reply** intelligently to important emails
- **Archive** spam and promo emails
- **Manage deadlines** — urgent emails expire if not handled in time

## Environment Design

| Component | Description |
|---|---|
| State | inbox, current_time, pending_tasks, completed_tasks |
| Actions | classify, prioritize, reply, archive, noop |
| Reward | Dense, step-level, range [-1.0, 1.0] |
| Max Steps | 25 per episode |

## Tasks

| Task | Emails | Difficulty | Expected Score |
|---|---|---|---|
| easy | 5 | Simple classification | 0.60 – 0.80 |
| medium | 10 | Classification + reply + mild ambiguity | 0.40 – 0.60 |
| hard | 15 | Deadlines + hidden priorities + ambiguity | 0.30 – 0.50 |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/tasks` | List available tasks |
| POST | `/reset` | Reset environment |
| POST | `/step` | Take an action |
| GET | `/state` | Get current state |
| POST | `/grade` | Grade completed episode |

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run server (set PORT to override the default 7860)
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the baseline agent against the server (heuristic — no API key needed)
export ENV_BASE_URL=http://localhost:7860
export TASK=easy                     # easy | medium | hard
python inference.py

# Optional: let an LLM supply triage hints (any OpenAI-compatible endpoint)
export LLM_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=your_key
export MODEL_NAME=gpt-4o-mini

# Run the self-check
python test_env.py

# Prove the LLM-call / latency optimization (offline, no API key)
python llm_agent.py

# Build the fine-tuning dataset from the env's ground truth (offline, no API key)
python finetune.py
```

## Deployment

One image runs everywhere. The container reads `PORT` (default `7860`), so
HuggingFace gets `7860` and Azure gets whatever port it injects — no rebuild.

### Docker (local / anywhere)
```bash
docker build -t exec-inbox .
docker run -p 7860:7860 exec-inbox              # HF-style default
docker run -e PORT=8080 -p 8080:8080 exec-inbox # custom port
```

### HuggingFace Spaces
Docker SDK Space, `app_port: 7860` (see the front-matter at the top of this
file). The optimized environment is deployed and running live here:
**[https://huggingface.co/spaces/praxshant/exec_inbox_env](https://huggingface.co/spaces/praxshant/exec_inbox_env)**

To deploy updates, push these files to the Space's git remote and it builds automatically:
```bash
git remote add hf https://huggingface.co/spaces/<user>/exec_inbox_env
git push hf main
```

### Azure — free tier (Docker)
The optimized environment is deployed and running live on Azure Container Apps (`eastus`):
**[https://exec-inbox.gentleisland-69027098.eastus.azurecontainerapps.io/](https://exec-inbox.gentleisland-69027098.eastus.azurecontainerapps.io/)**

*(Note: Ingress is external, the API is public but holds no secrets/state per session.)*

Use **Azure Container Apps** (consumption plan): first 180,000 vCPU-s,
360,000 GiB-s, and 2M requests per month are free, and it scales to zero when
idle. Push the image to any registry (Docker Hub free works), then:
```bash
az containerapp up \
  --name exec-inbox --resource-group exec-inbox-rg \
  --image docker.io/<user>/exec-inbox:latest \
  --ingress external --target-port 7860
```
Container Apps routes ingress to `--target-port`, so the default `7860` is
fine; no `PORT` override needed.

> App Service **F1 (free)** does *not* run custom Linux containers — that needs
> Basic (B1+). For strictly-free Docker on Azure, use Container Apps above.

## Baseline Scores

Heuristic agent (no LLM), measured via `inference.py` and `test_env.py`:

| Task | Score |
|---|---|
| easy | ~0.78 |
| medium | ~0.63 |
| hard | ~0.64 |

## LLM Optimization & Fine-Tuning

The LLM path is built for **low latency and low cost**, and the win is proven,
not asserted. Run `python llm_agent.py` to reproduce the table below.

**How it's optimized:**
- **One call per episode, not one per step.** The whole inbox is triaged in a
  single structured call; the agent then executes that plan across all steps
  with zero further calls. This is the big lever — it removes an O(steps)
  network cost.
- **Structured JSON output** (`response_format=json_object`, `temperature=0`) —
  deterministic, no re-prompting or parse-retry round-trips.
- **Compact prompt** — only `id/sender/subject/body[:200]` per email; bodies
  truncated → fewer input tokens.
- **Signature cache** — identical inboxes reuse the plan for free (0 calls).
- **Heuristic fallback** — with no API key the same code path runs on rules at
  zero latency, so the environment never hard-depends on an LLM.

**Proof** (mock LLM at a fixed 0.15 s/call round-trip; *same policy both ways*,
so score is identical and only efficiency moves):

| task | strategy | LLM calls | input tokens | wall time | score |
|---|---|---|---|---|---|
| easy | per-step (old) | 13 | 2,524 | 3.30 s | 0.927 |
| easy | **batched (new)** | **1** | **292** | **0.15 s** | 0.927 |
| medium | per-step (old) | 24 | 7,019 | 3.62 s | 0.780 |
| medium | **batched (new)** | **1** | **546** | **0.15 s** | 0.780 |
| hard | per-step (old) | 25 | 12,727 | 3.77 s | 0.713 |
| hard | **batched (new)** | **1** | **807** | **0.15 s** | 0.713 |

→ **13–25× fewer calls, ~9–16× fewer input tokens, ~22–25× lower latency, zero
score loss.**

**Fine-tuning** (`python finetune.py`): the env already holds the correct
label/priority for every email, so `finetune.py` emits an OpenAI-format
`finetune.jsonl` (33 examples) in the *exact* single-call format above — free
supervised data. After `python finetune.py --launch` (needs `OPENAI_API_KEY`),
point `MODEL_NAME` at the tuned model. A fine-tuned model learns the schema, so
you can drop the few-shot scaffolding from the prompt → **shorter prompts +
smaller model = another latency/cost cut** on top of the batching win.

## Reward Design

- `+0.2` correct classification
- `+0.3` correct priority
- `+0.4` reply to high-priority email
- `+0.2` archive spam/promo
- `-0.4` reply to spam
- `-0.3` miss high-priority deadline
- `-0.15` archive important email