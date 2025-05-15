import asyncio
from core import Browser


async def producer(q: asyncio.Queue[str]) -> None:
    loop = asyncio.get_running_loop()
    while True:
        cmd = await loop.run_in_executor(None, input, "Enter id (empty to quit): ")
        await q.put(cmd.strip())


async def main() -> None:
    events: asyncio.Queue[str] = asyncio.Queue()
    asyncio.create_task(producer(events))

    browser = Browser()
    await browser.start()
    await browser.goto("https://www.google.com")
    await browser.enable_overlay()

    while True:
        command = await events.get()

        if not command:       # empty → exit
            break
        if not command.isdigit():
            print("⛔  please type a numeric id")
            continue

        try:
            await browser.click(int(command))
            print("✓ clicked", command)
        except Exception as e:
            print("⚠️ ", e)

    await browser.close()

asyncio.run(main())
