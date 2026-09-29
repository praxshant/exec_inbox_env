# /server/app.py

import os

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict

from env.environment import EmailEnv
from env.models import Action
from env.grader import grade_episode
from env.tasks import TASKS

app = FastAPI(
    title="ExecInbox — AI Executive Assistant Environment",
    description="OpenEnv-compatible inbox management simulation for AI agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

envs: Dict[str, EmailEnv] = {
    "easy": EmailEnv(task="easy"),
    "medium": EmailEnv(task="medium"),
    "hard": EmailEnv(task="hard"),
}

# Initialize all envs on startup
for task_name, env in envs.items():
    env.reset()


class ActionRequest(BaseModel):
    type: str
    email_id: Optional[str] = None
    label: Optional[str] = None
    priority: Optional[str] = None
    content: Optional[str] = None


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


# Optional API-key auth. If API_KEY is set in the environment, the mutating and
# state endpoints require it (header `X-API-Key: <key>` or `Authorization:
# Bearer <key>`). If API_KEY is unset the API is open — the public demo runs
# open on purpose: it holds no secrets and no cross-session state. Health (`/`)
# and task discovery (`/tasks`) stay open either way.
def require_key(
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
):
    expected = os.environ.get("API_KEY")
    if not expected:
        return
    token = x_api_key or (authorization or "").removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def health():
    return {"status": "ok", "environment": "ExecInbox", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": list(TASKS.keys()),
        "description": {
            "easy": "5 emails, simple classification with light deadlines",
            "medium": "10 emails, classification + reply, mild ambiguity",
            "hard": "15 emails, deadlines + priorities + ambiguous intent",
        },
    }


@app.post("/reset", dependencies=[Depends(require_key)])
def reset(
    req: Optional[ResetRequest] = None,
    task: str = Query(default="easy"),
    seed: int = Query(default=42),
):
    # Body takes priority over query params
    actual_task = req.task if req else task
    actual_seed = req.seed if req else seed

    if actual_task not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task. Choose from: {list(TASKS.keys())}",
        )

    envs[actual_task] = EmailEnv(task=actual_task)
    obs = envs[actual_task].reset(seed=actual_seed)
    return obs.model_dump()


@app.post("/step", dependencies=[Depends(require_key)])
def step(action: ActionRequest, task: str = Query(default="easy")):
    if task not in envs:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task. Choose from: {list(TASKS.keys())}",
        )
    env = envs[task]
    action_obj = Action(**action.model_dump())
    try:
        obs, reward, done, info = env.step(action_obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state", dependencies=[Depends(require_key)])
def state(task: str = Query(default="easy")):
    if task not in envs:
        raise HTTPException(status_code=400, detail="Invalid task.")
    return envs[task].state().model_dump()


@app.post("/grade", dependencies=[Depends(require_key)])
def grade(task: str = Query(default="easy")):
    if task not in envs:
        raise HTTPException(status_code=400, detail="Invalid task.")
    env = envs[task]
    ground_truth = TASKS[task]
    actions = env.get_action_history()
    scores = grade_episode(actions, ground_truth)
    return scores

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()