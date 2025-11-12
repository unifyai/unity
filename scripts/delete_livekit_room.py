import asyncio
from dotenv import load_dotenv

load_dotenv()
from livekit import api


async def terminate_room(room_name: str):
    lkapi = api.LiveKitAPI()

    try:
        rooms = await lkapi.room.list_rooms(api.ListRoomsRequest())
        room_names = [room.name for room in rooms.rooms]
        print(room_names)

        if room_name in room_names:
            remove_req = api.DeleteRoomRequest(room=room_name)
            await lkapi.room.delete_room(remove_req)
            print(f"Successfully terminated room {room_name}.")
        else:
            print(f"Room {room_name} not found in any active rooms.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await lkapi.aclose()


if __name__ == "__main__":
    asyncio.run(terminate_room("unity_"))
