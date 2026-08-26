"""Conversation transcript construction for the multi-turn chat workload.

A one-shot request sends an independent prompt. A multi-turn request sends the
*whole transcript so far* — system prompt, every prior user line, and the
model's own prior replies — plus one new user line. The transcript grows every
turn, so the prefix a KV-cache can reuse grows with it. See
docs/superpowers/specs/2026-08-26-phase7-multiturn-chat.md.
"""

from __future__ import annotations


class Conversation:
    """Accumulates a chat transcript one turn at a time.

    `prompt_for(user_turn)` returns the prompt to send for that turn without
    mutating anything; `commit(user_turn, reply)` folds the turn (and the
    model's actual reply) into the transcript so the next `prompt_for` includes
    it. Splitting the two lets a benchmark run the prompt through more than one
    code path before deciding which reply is canonical.
    """

    def __init__(self, system: str) -> None:
        self.transcript = system.rstrip()

    def prompt_for(self, user_turn: str) -> str:
        return f"{self.transcript}\nUser: {user_turn}\nAssistant:"

    def commit(self, user_turn: str, assistant_reply: str) -> None:
        self.transcript = f"{self.prompt_for(user_turn)} {assistant_reply.strip()}"


def build_multiturn_prompts(
    system: str, user_turns: list[str], replies: list[str]
) -> list[str]:
    """Batch form of the incremental `Conversation` above, for when every reply
    is already known (tests, plotting fixtures).

    `replies` may be one shorter than `user_turns` (the last turn has not been
    answered yet). Returns one prompt per user turn.
    """
    if len(replies) not in (len(user_turns), len(user_turns) - 1):
        raise ValueError(
            f"replies must have len(user_turns) or len(user_turns)-1 entries, "
            f"got {len(replies)} for {len(user_turns)} turns"
        )
    conversation = Conversation(system)
    prompts: list[str] = []
    for k, user_turn in enumerate(user_turns):
        prompts.append(conversation.prompt_for(user_turn))
        if k < len(replies):
            conversation.commit(user_turn, replies[k])
    return prompts
