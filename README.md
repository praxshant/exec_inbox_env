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

# Run server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run baseline inference (set env vars first)
export OPENAI_API_KEY=your_key
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=gpt-4o-mini
python inference.py
```

## Docker
```bash
docker build -t exec-inbox .
docker run -p 7860:7860 exec-inbox
```

## Baseline Scores

| Task | Score |
|---|---|
| easy | ~0.70 |
| medium | ~0.50 |
| hard | ~0.38 |

## Reward Design

- `+0.2` correct classification
- `+0.3` correct priority
- `+0.4` reply to high-priority email
- `+0.2` archive spam/promo
- `-0.4` reply to spam
- `-0.3` miss high-priority deadline
- `-0.15` archive important email