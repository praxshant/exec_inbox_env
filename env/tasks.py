# env/tasks.py

from typing import Dict, List
from env.models import Email


def get_easy_task() -> List[Email]:
    """
    5 emails. Simple classification only.
    Labels are fairly obvious from content.
    No tricky deadlines — agent just needs to classify correctly.
    """
    return [
        Email(
            id="e1",
            sender="boss@company.com",
            subject="Meeting at 3 PM",
            body="Please attend the standup meeting at 3 PM today. Your presence is mandatory.",
            label="important",
            priority="high",
            deadline=3,
        ),
        Email(
            id="e2",
            sender="newsletter@shopping.com",
            subject="Big Sale Today! 50% OFF Everything",
            body="Don't miss our mega sale. Flat 50% off on all items today only.",
            label="promo",
            priority="low",
        ),
        Email(
            id="e3",
            sender="lottery@winner123.com",
            subject="You won $1,000,000!",
            body="Congratulations! You have been selected. Click here to claim your prize now.",
            label="spam",
            priority="low",
        ),
        Email(
            id="e4",
            sender="hr@company.com",
            subject="Updated Leave Policy — Action Required",
            body="Please review and acknowledge the updated HR leave policy by end of week.",
            label="important",
            priority="medium",
            deadline=5,
        ),
        Email(
            id="e5",
            sender="friend@gmail.com",
            subject="Weekend hiking trip?",
            body="Hey! Are we still planning the hiking trip this weekend? Let me know.",
            label="important",
            priority="low",
        ),
    ]


def get_medium_task() -> List[Email]:
    """
    10 emails. Classification + reply + mild ambiguity.
    Some emails look important but are promo. Agent must reason.
    """
    return [
        Email(
            id="m1",
            sender="client@bigcorp.com",
            subject="Critical: Product Down in Production",
            body="Our entire team is blocked. The product stopped working 2 hours ago. Need urgent resolution.",
            label="important",
            priority="high",
            deadline=2,
        ),
        Email(
            id="m2",
            sender="deals@bestbuy.com",
            subject="Flash Sale — Laptops from $299",
            body="Limited time deal. Grab the best laptops at unbeatable prices.",
            label="promo",
            priority="low",
        ),
        Email(
            id="m3",
            sender="security-noreply@paypa1.com",
            subject="Your account has been suspended",
            body="Verify your PayPal account immediately or it will be permanently closed.",
            label="spam",
            priority="low",
        ),
        Email(
            id="m4",
            sender="manager@company.com",
            subject="Project Status Update Needed",
            body="I need the full project status report before our 4 PM call today. Please send ASAP.",
            label="important",
            priority="high",
            deadline=3,
        ),
        Email(
            id="m5",
            sender="devteam@company.com",
            subject="PR #247 — Needs Your Review",
            body="Hey, we need your approval on PR #247 before we can merge and deploy. Deadline is EOD.",
            label="important",
            priority="medium",
            deadline=4,
        ),
        Email(
            id="m6",
            sender="events@techcommunity.com",
            subject="Free Webinar: Boost Your Productivity",
            body="Join our free webinar on productivity hacks for engineers. Register now.",
            label="promo",
            priority="low",
        ),
        Email(
            id="m7",
            sender="support@saasproduct.com",
            subject="Your Support Ticket #8821 Updated",
            body="We have updated your ticket. The issue is now being escalated to our senior team.",
            label="important",
            priority="medium",
        ),
        Email(
            id="m8",
            sender="cfo@company.com",
            subject="Budget Approval Pending",
            body="The Q3 budget proposal needs your sign-off before I can proceed with vendor payments.",
            label="important",
            priority="high",
            deadline=2,
        ),
        Email(
            id="m9",
            sender="noreply@linkedin.com",
            subject="You have 5 new connection requests",
            body="People are waiting to connect with you on LinkedIn. Accept requests now.",
            label="promo",
            priority="low",
        ),
        Email(
            id="m10",
            sender="colleague@company.com",
            subject="Quick question about the API",
            body="Hey, do you know if the new API endpoint supports pagination? Just checking before I build.",
            label="important",
            priority="low",
        ),
    ]


def get_hard_task() -> List[Email]:
    """
    15 emails. Deadlines, hidden priorities, ambiguity, trade-offs.
    Some 'important' senders are sending promo. Some spam looks real.
    Agent must infer priority from context, not just sender/subject.
    """
    return [
        Email(
            id="h1",
            sender="ceo@company.com",
            subject="URGENT: Investor Deck — Needs to be Ready",
            body="The board meeting is at 2 PM. I need the investor presentation ready with Q3 numbers. No delays.",
            label="important",
            priority="high",
            deadline=1,
        ),
        Email(
            id="h2",
            sender="client@enterprise.com",
            subject="Following up",
            body="We still haven't heard back. This is becoming critical for our team.",
            label="important",
            priority="high",
            deadline=2,
        ),
        Email(
            id="h3",
            sender="security@bank-alerts.com",
            subject="Unusual activity detected",
            body="We noticed something unusual. Please review your account activity.",
            label="important",
            priority="high",
            deadline=2,
        ),
        Email(
            id="h4",
            sender="newsletter@amazon.com",
            subject="Your order has shipped!",
            body="Great news — your Amazon order #112-4432 has shipped and will arrive by Friday.",
            label="promo",
            priority="low",
        ),
        Email(
            id="h5",
            sender="winner@prize-claim.net",
            subject="You've been selected for a reward",
            body="Our system has selected your email address for a special reward. Click to claim before midnight.",
            label="spam",
            priority="low",
        ),
        Email(
            id="h6",
            sender="hr@company.com",
            subject="Interview Panel — Confirm Availability",
            body="You're scheduled to interview a candidate tomorrow at 11 AM. Please confirm your availability.",
            label="important",
            priority="medium",
            deadline=4,
        ),
        Email(
            id="h7",
            sender="scrum@company.com",
            subject="Sprint Planning Tomorrow — Agenda Attached",
            body="Sprint planning is tomorrow morning at 9 AM. Please review the backlog before joining.",
            label="important",
            priority="medium",
            deadline=3,
        ),
        Email(
            id="h8",
            sender="friend@gmail.com",
            subject="Finalizing the Goa trip",
            body="Hey! We need to book the hotel by tonight or prices go up. Can you confirm if you're in?",
            label="important",
            priority="low",
        ),
        Email(
            id="h9",
            sender="security@github.com",
            subject="Action required: SSH key expiring",
            body="One of your SSH keys will expire in 24 hours. Please rotate it to avoid disruption.",
            label="important",
            priority="medium",
            deadline=3,
        ),
        Email(
            id="h10",
            sender="updates@notion.so",
            subject="What's new in Notion this month",
            body="Check out the latest features we've shipped — databases, AI blocks, and more.",
            label="promo",
            priority="low",
        ),
        Email(
            id="h11",
            sender="legal@company.com",
            subject="Contract Review — Sign Before Friday",
            body="The vendor contract needs your review and e-signature before Friday or the deal lapses.",
            label="important",
            priority="high",
            deadline=4,
        ),
        Email(
            id="h12",
            sender="noreply@medium.com",
            subject="Your weekly reading list is ready",
            body="Based on your interests, here are 5 stories we think you'll enjoy this week.",
            label="promo",
            priority="low",
        ),
        Email(
            id="h13",
            sender="ops@company.com",
            subject="Server utilization at 94% — Action Needed",
            body="Production server CPU has been above 90% for the past hour. Please investigate before it impacts users.",
            label="important",
            priority="high",
            deadline=2,
        ),
        Email(
            id="h14",
            sender="refund@shop-deals.biz",
            subject="Your refund of $500 is pending — verify now",
            body="To process your refund, click the link and enter your bank details for verification.",
            label="spam",
            priority="low",
        ),
        Email(
            id="h15",
            sender="mentor@university.edu",
            subject="Research paper feedback",
            body="I've reviewed your paper draft. There are some significant revisions needed before submission next week.",
            label="important",
            priority="medium",
            deadline=5,
        ),
    ]


TASKS: Dict[str, List[Email]] = {
    "easy": get_easy_task(),
    "medium": get_medium_task(),
    "hard": get_hard_task(),
}