import sys, pathlib
import asyncio

# Ensure repository root is on PYTHONPATH so `import unity` works when this
# script is executed directly from inside the "sandboxes" folder.
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from unity.controller.controller import Controller

async def main():
    # Instantiate and start the controller thread
    controller = Controller()
    controller.start()

    while True:
        act_or_observe = input("Act or Observe? (a/o/c): ")
        if act_or_observe == "a":
            print(controller._observe_ctx)
            action = input("Action: ")
            act_res = await controller.act(action)
            print("Action result:", act_res)
        elif act_or_observe == "o":
            obs = input("Observation: ")
            obs_res = await controller.observe(obs, bool)
            print("Observe result:", obs_res)
        elif act_or_observe == "c":
            act_res = await controller.act("close_browser")
            print("Action result:", act_res)
            break
        else:
            print("Invalid method")

if __name__ == "__main__":
    asyncio.run(main())
