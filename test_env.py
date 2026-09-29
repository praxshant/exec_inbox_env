# test_env.py — runnable self-check for reward logic, grading, the heuristic
# agent, and the HTTP layer. No framework: `python test_env.py`.

from env.environment import EmailEnv
from env.grader import grade_episode
from env.models import Action
from env.tasks import TASKS
from inference import smart_action


def test_reward_paths():
    env = EmailEnv("easy")
    env.reset()
    # e3 is lottery spam
    _, r, _, _ = env.step(Action(type="classify", email_id="e3", label="spam"))
    assert "correct_classification" in r.reason and r.value > 0, r
    # e2 is promo, not spam
    _, r, _, _ = env.step(Action(type="classify", email_id="e2", label="spam"))
    assert "wrong_classification" in r.reason, r
    # re-classifying e3 is a duplicate
    _, r, _, _ = env.step(Action(type="classify", email_id="e3", label="spam"))
    assert "duplicate_classification" in r.reason, r
    # replying to spam is punished
    _, r, _, _ = env.step(Action(type="reply", email_id="e3", content="hello"))
    assert "replied_to_spam" in r.reason and r.value < 0, r
    # e1 (boss) is high priority
    _, r, _, _ = env.step(Action(type="prioritize", email_id="e1", priority="high"))
    assert "correct_priority" in r.reason, r


def test_deadline_penalty():
    env = EmailEnv("easy")
    env.reset()  # e1 has deadline=3
    reasons = []
    for _ in range(5):
        _, r, _, _ = env.step(Action(type="noop"))
        reasons.append(r.reason)
    assert any("deadline_penalty" in x for x in reasons), reasons


def test_agent_and_grader():
    print("scores:")
    for task in TASKS:
        env = EmailEnv(task)
        env.reset()
        replied, classified, prioritized, handled = set(), set(), set(), set()
        for _ in range(env.max_steps):
            obs = env.state().model_dump()
            act = smart_action(obs, replied, classified, prioritized, handled)
            _, _, done, _ = env.step(Action(**act))
            if done:
                break
        scores = grade_episode(env.get_action_history(), TASKS[task])
        print(f"  {task}: {scores}")
        assert 0.0 <= scores["final_score"] <= 1.0, scores
        assert scores["final_score"] >= 0.3, f"{task} regressed: {scores}"


def test_http_layer():
    from fastapi.testclient import TestClient
    from server.app import app
    c = TestClient(app)
    assert c.get("/").json()["status"] == "ok"
    assert "easy" in c.get("/tasks").json()["tasks"]
    r = c.post("/reset", json={"task": "easy", "seed": 42}).json()
    assert len(r["inbox"]) == 5, r
    step = c.post("/step", params={"task": "easy"},
                  json={"type": "classify", "email_id": "e3", "label": "spam"}).json()
    assert step["reward"]["value"] > 0, step
    assert "final_score" in c.post("/grade", params={"task": "easy"}).json()


def test_auth():
    # When API_KEY is set, health/tasks stay open but mutating/state endpoints
    # require the key; when unset the whole API is open.
    import os
    from fastapi.testclient import TestClient
    from server.app import app
    c = TestClient(app)
    os.environ["API_KEY"] = "secret123"
    try:
        assert c.get("/").status_code == 200                       # health open
        assert c.get("/tasks").status_code == 200                  # discovery open
        assert c.post("/reset", json={"task": "easy"}).status_code == 401
        r = c.post("/reset", json={"task": "easy"},
                   headers={"X-API-Key": "secret123"})
        assert r.status_code == 200, r.text
        r = c.post("/reset", json={"task": "easy"},
                   headers={"Authorization": "Bearer secret123"})
        assert r.status_code == 200, r.text
    finally:
        os.environ.pop("API_KEY", None)


if __name__ == "__main__":
    test_reward_paths()
    test_deadline_penalty()
    test_agent_and_grader()
    test_http_layer()
    test_auth()
    print("OK — all checks passed")
