# env/grader.py

from typing import List, Dict
from env.models import Email, Action


def grade_episode(actions: List[Action], ground_truth: List[Email]) -> Dict[str, float]:
    """
    Episode-level grader. Returns component scores and final score.
    Score range: 0.0 to 1.0

    Components:
        - classification_score (0.2 weight)
        - priority_score       (0.3 weight)
        - deadline_score       (0.3 weight)
        - reply_score          (0.2 weight)
    """
    email_map: Dict[str, Email] = {e.id: e for e in ground_truth}

    # Track per-component hits
    classify_correct = 0
    classify_total = 0
    priority_correct = 0
    priority_total = 0
    reply_score_sum = 0.0
    reply_total = 0
    deadline_met = 0
    deadline_total = 0

    # Track which emails were acted on and when
    email_action_time: Dict[str, int] = {}  # email_id -> step when actioned

    penalties = 0.0

    for step_idx, action in enumerate(actions):
        email = email_map.get(action.email_id) if action.email_id else None

        if action.type == "classify" and email:
            classify_total += 1
            if action.label == email.label:
                classify_correct += 1

        elif action.type == "prioritize" and email:
            priority_total += 1
            if action.priority == email.priority:
                priority_correct += 1
            # Partial credit for adjacent
            elif (action.priority, email.priority) in {
                ("high", "medium"), ("medium", "high"),
                ("medium", "low"), ("low", "medium"),
            }:
                priority_correct += 0.5

        elif action.type == "reply" and email:
            reply_total += 1
            if email.label == "spam":
                penalties += 0.5  # heavy penalty
            elif email.label == "promo":
                penalties += 0.2
            else:
                # Quality based on priority
                if email.priority == "high":
                    reply_score_sum += 1.0
                elif email.priority == "medium":
                    reply_score_sum += 0.6
                else:
                    reply_score_sum += 0.3

                # Deadline check
                if email.deadline is not None:
                    deadline_total += 1
                    if step_idx + 1 <= email.deadline:
                        deadline_met += 1

        elif action.type == "archive" and email:
            if email.label == "important" and email.priority == "high":
                penalties += 0.4

    # Emails with deadlines that were never actioned
    actioned_ids = {a.email_id for a in actions if a.email_id}
    for email in ground_truth:
        if email.deadline is not None and email.id not in actioned_ids:
            deadline_total += 1
            penalties += 0.3 if email.priority == "high" else 0.1

    # Component scores 
    classification_score = (
        classify_correct / classify_total if classify_total > 0 else 0.0
    )
    priority_score = (
        priority_correct / priority_total if priority_total > 0 else 0.0
    )
    reply_score = (
        reply_score_sum / reply_total if reply_total > 0 else 0.0
    )
    deadline_score = (
        deadline_met / deadline_total if deadline_total > 0 else 1.0
    )

    # Weighted final score
    raw_score = (
        0.2 * classification_score
        + 0.3 * priority_score
        + 0.3 * deadline_score
        + 0.2 * reply_score
    )

    # penalize too many actions
    if len(actions) > 20:
        penalties += 0.2

    # Apply penalties (capped at 0.5 total penalty)
    penalty_deduction = min(penalties * 0.1, 0.5)
    final_score = max(0.0, min(1.0, raw_score - penalty_deduction))

    return {
        "final_score": round(final_score, 4),
        "classification_score": round(classification_score, 4),
        "priority_score": round(priority_score, 4),
        "deadline_score": round(deadline_score, 4),
        "reply_score": round(reply_score, 4),
        "penalties": round(penalty_deduction, 4),
    }