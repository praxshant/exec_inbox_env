# env/environment.py

import random
from typing import Tuple, Dict, Any, List, Set

from env.models import Observation, Action, Reward, Email, Task, PublicEmail
from env.tasks import TASKS


class EmailEnv:
    def __init__(self, task: str = "easy"):
        if task not in TASKS:
            raise ValueError(f"Invalid task '{task}'. Choose from: {list(TASKS.keys())}")

        self.task_name = task
        self.emails: List[Email] = []
        self.current_time: int = 0
        self.pending_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.done: bool = False
        self.max_steps: int = 25
        self.steps: int = 0
        self._action_history: List[Action] = []
        self._penalized_deadlines: Set[str] = set()

    def reset(self, seed: int = 42) -> Observation:
        random.seed(seed)
        self.emails = [email.copy(deep=True) for email in TASKS[self.task_name]]
        self.current_time = 0
        self.pending_tasks = []
        self.completed_tasks = []
        self.done = False
        self.steps = 0
        self._action_history = []
        self._penalized_deadlines = set()
        return self.state()

    def state(self) -> Observation:
        public_inbox = [
            PublicEmail(
                id=e.id,
                sender=e.sender,
                subject=e.subject,
                body=e.body,
            )
            for e in self.emails
        ]
        return Observation(
            inbox=public_inbox,
            current_time=self.current_time,
            pending_tasks=self.pending_tasks,
            completed_tasks=self.completed_tasks,
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        try:
            if self.done:
                return (
                    self.state(),
                    Reward(value=0.0, reason="episode_already_done"),
                    True,
                    {},
                )

            self.steps += 1
            self.current_time += 1
            self._action_history.append(action)

            reward_value = -0.02  # time cost
            reason = "neutral"

            if action.type == "noop":
                reward_value = -0.05
                reason = "noop_penalty"

            elif action.type == "classify":
                r, reason = self._handle_classify(action)
                reward_value += r

            elif action.type == "prioritize":
                r, reason = self._handle_prioritize(action)
                reward_value += r

            elif action.type == "reply":
                r, reason = self._handle_reply(action)
                reward_value += r

            elif action.type == "archive":
                r, reason = self._handle_archive(action)
                reward_value += r

            else:
                reward_value = -0.2
                reason = "unknown_action_type"

            deadline_penalty = self._check_deadlines()
            reward_value += deadline_penalty
            if deadline_penalty < 0:
                reason += f"|deadline_penalty({deadline_penalty:.2f})"

            if self.steps >= self.max_steps or len(self.emails) == 0:
                self.done = True

            reward = Reward(
                value=max(-1.0, min(1.0, reward_value)),
                reason=reason,
            )

            return self.state(), reward, self.done, {"steps": self.steps}

        except Exception as e:
            self.done = True
            return (
                self.state(),
                Reward(value=-1.0, reason=f"env_error: {str(e)}"),
                True,
                {"error": str(e)},
            )

    def _get_email(self, email_id: str):
        for email in self.emails:
            if email.id == email_id:
                return email
        return None

    def _handle_classify(self, action: Action) -> Tuple[float, str]:
        if not action.email_id or not action.label:
            return -0.2, "invalid_classify_missing_fields"

        email = self._get_email(action.email_id)
        if not email:
            return -0.2, "classify_email_not_found"

        # Prevent duplicate classification
        for prev in self._action_history[:-1]:
            if prev.type == "classify" and prev.email_id == email.id:
                return -0.1, "duplicate_classification"

        if action.label == email.label:
            return 0.2, "correct_classification"

        return -0.1, f"wrong_classification(got={action.label},expected={email.label})"

    def _handle_prioritize(self, action: Action) -> Tuple[float, str]:
        if not action.email_id or not action.priority:
            return -0.2, "invalid_prioritize_missing_fields"

        email = self._get_email(action.email_id)
        if not email:
            return -0.2, "prioritize_email_not_found"

        # Prevent duplicate prioritization
        for prev in self._action_history[:-1]:
            if prev.type == "prioritize" and prev.email_id == email.id:
                return -0.1, "duplicate_prioritize"

        if action.priority == email.priority:
            return 0.3, "correct_priority"

        adjacent = {
            ("high", "medium"), ("medium", "high"),
            ("medium", "low"), ("low", "medium"),
        }
        if (action.priority, email.priority) in adjacent:
            return 0.1, f"adjacent_priority(got={action.priority},expected={email.priority})"

        return -0.15, f"wrong_priority(got={action.priority},expected={email.priority})"

    def _handle_reply(self, action: Action) -> Tuple[float, str]:
        if not action.email_id or not action.content or not action.content.strip():
            return -0.2, "invalid_reply_missing_fields"

        email = self._get_email(action.email_id)
        if not email:
            return -0.2, "reply_email_not_found"

        if email.label == "spam":
            return -0.4, "replied_to_spam"
        if email.label == "promo":
            return -0.2, "replied_to_promo"

        if email.priority == "high":
            reward = 0.4
        elif email.priority == "medium":
            reward = 0.25
        else:
            reward = 0.1

        # Semantic bonus
        content_lower = action.content.lower()
        if "confirm" in content_lower or "will do" in content_lower:
            reward += 0.05
        if "thanks" in content_lower or "thank you" in content_lower:
            reward += 0.05

        # Deadline bonus
        if email.deadline is not None and self.current_time <= email.deadline:
            reward += 0.1

        self.emails = [e for e in self.emails if e.id != email.id]

        if email.deadline:
            task = Task(
                id=f"task_{email.id}",
                description=f"Replied to: {email.subject}",
                deadline=email.deadline,
                completed=True,
            )
            self.completed_tasks.append(task)

        return min(reward, 0.5), "reply_sent"

    def _handle_archive(self, action: Action) -> Tuple[float, str]:
        if not action.email_id:
            return -0.2, "invalid_archive_missing_fields"

        email = self._get_email(action.email_id)
        if not email:
            return -0.2, "archive_email_not_found"

        if email.label in ["spam", "promo"]:
            self.emails = [e for e in self.emails if e.id != email.id]
            return 0.2, "correctly_archived_spam_or_promo"

        penalty = -0.3 if email.priority == "high" else -0.15
        self.emails = [e for e in self.emails if e.id != email.id]
        return penalty, f"wrongly_archived_important(priority={email.priority})"

    def _check_deadlines(self) -> float:
        penalty = 0.0
        for email in self.emails:
            if (
                email.deadline is not None
                and self.current_time > email.deadline
                and email.id not in self._penalized_deadlines
            ):
                if email.priority == "high":
                    penalty -= 0.4
                elif email.priority == "medium":
                    penalty -= 0.2
                else:
                    penalty -= 0.1
                self._penalized_deadlines.add(email.id)
        return penalty

    def get_action_history(self) -> List[Action]:
        return self._action_history