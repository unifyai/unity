import json

from unity.common.context_dump import make_messages_safe_for_context_dump


def test_make_messages_safe_for_context_dump_redacts_image_blobs() -> None:
    raw_data_url = f"data:image/png;base64,{'A' * 4000}"
    raw_screenshot = "B" * 1200
    messages = [
        {
            "role": "tool",
            "name": "execute_code",
            "content": [
                {"type": "text", "text": "captured screenshot"},
                {"type": "image_url", "image_url": {"url": raw_data_url}},
                {"type": "text", "text": f"inline url: {raw_data_url}"},
            ],
            "computer_state": {
                "url": "https://example.com",
                "screenshot": raw_screenshot,
            },
        },
    ]

    safe = make_messages_safe_for_context_dump(messages)
    serialized = json.dumps(safe)

    assert raw_data_url not in serialized
    assert raw_screenshot not in serialized
    assert "data:image/png;base64,<omitted>" in serialized
    assert "<image omitted>" in serialized

    # Ensure original payload is unchanged (deep-copy semantics).
    assert messages[0]["computer_state"]["screenshot"] == raw_screenshot
    assert messages[0]["content"][1]["image_url"]["url"] == raw_data_url


def test_make_messages_safe_for_context_dump_handles_none() -> None:
    assert make_messages_safe_for_context_dump(None) == []
