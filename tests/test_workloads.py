import pytest

from cachequant.workloads import Conversation, build_multiturn_prompts

SYSTEM = "You are a helpful assistant."
TURNS = ["What is the capital of France?", "What river runs through it?", "How old is it?"]
REPLIES = ["Paris.", "The Seine.", "About 2000 years."]


def test_prompt_for_format():
    conv = Conversation(SYSTEM)
    assert conv.prompt_for("Hi there") == f"{SYSTEM}\nUser: Hi there\nAssistant:"


def test_commit_grows_transcript_and_keeps_prefix():
    conv = Conversation(SYSTEM)
    first = conv.prompt_for(TURNS[0])
    conv.commit(TURNS[0], REPLIES[0])
    second = conv.prompt_for(TURNS[1])

    # Turn 2's prompt is turn 1's prompt + " <reply>" + the new user line.
    assert second.startswith(first + f" {REPLIES[0]}")
    assert second.endswith(f"\nUser: {TURNS[1]}\nAssistant:")


def test_prompt_length_strictly_increases():
    conv = Conversation(SYSTEM)
    lengths = []
    for turn, reply in zip(TURNS, REPLIES):
        lengths.append(len(conv.prompt_for(turn)))
        conv.commit(turn, reply)
    assert lengths == sorted(lengths)
    assert len(set(lengths)) == len(lengths)


def test_commit_strips_reply_whitespace():
    conv = Conversation(SYSTEM)
    conv.commit(TURNS[0], "  Paris.  ")
    assert conv.transcript.endswith("Assistant: Paris.")


def test_batch_builder_matches_incremental():
    incremental = []
    conv = Conversation(SYSTEM)
    for turn, reply in zip(TURNS, REPLIES):
        incremental.append(conv.prompt_for(turn))
        conv.commit(turn, reply)

    assert build_multiturn_prompts(SYSTEM, TURNS, REPLIES) == incremental


def test_batch_builder_allows_one_fewer_reply_than_turns():
    prompts = build_multiturn_prompts(SYSTEM, TURNS, REPLIES[:-1])
    assert len(prompts) == len(TURNS)


def test_batch_builder_rejects_mismatched_reply_count():
    with pytest.raises(ValueError):
        build_multiturn_prompts(SYSTEM, TURNS, REPLIES[:1])
