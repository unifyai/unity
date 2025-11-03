import asyncio
import json


async def capture_stream_response(pubsub, label: str, timeout: float = 60.0):
    """Capture start_gen -> chunks -> end_gen"""
    chunks = []
    start_time = asyncio.get_event_loop().time()
    got_start = False
    got_end = False

    while (asyncio.get_event_loop().time() - start_time) < timeout:
        try:
            msg = await asyncio.wait_for(
                pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                timeout=5.0,
            )

            if msg and msg["type"] == "message":
                data = json.loads(msg["data"])

                if data["type"] == "start_gen":
                    got_start = True
                    print(f"   ✓ {label}: Got start_gen")

                elif data["type"] == "gen_chunk":
                    chunk_content = data.get("chunk", "")
                    chunks.append(chunk_content)
                    print(f"   ✓ {label}: Got chunk: {chunk_content}")

                elif data["type"] == "end_gen":
                    got_end = True
                    full_response = "".join(chunks)
                    print(
                        f"   ✓ {label}: Got {len(chunks)} chunks, {len(full_response)} chars total"
                    )
                    print(f"   ✓ {label}: Preview: {full_response[:80]}...")
                    return got_start, chunks, got_end

        except asyncio.TimeoutError:
            continue

    return got_start, chunks, got_end
