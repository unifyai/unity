To run this demo, just run `main.py`, make sure to have `textual` and `textual-dev` installed.
This will run `events_manager.py` under-the-hood and `call.py` if you initiate a call.

To have better visibility, run the files in the following orders:
- `events_manager.py`
- `main.py`

`events_manager.py` will print the events stream, which is useful to understand what is going on.

At the moment the demo is basically just showcasing the idea of being able to "chat" across multiple platforms, switch to a call, continue chatting while calling, e.g sending a message to the assistant while on a call or recieving messages.

Tips:
- Do not run this within the vscode terminal, because it uses `textual`, closing `textual` apps is done using `ctrl+q` (or whatever is eqv in mac), which is a hot-key in vscode, instead just open a terminal window and run it there, (i can add a key binding for ctrl+c to close it, but until then do this).
- End calls using `End Call` button in the gui
- If launching a call fails let me know, i tested  this on windows, so im not sure if launching terminals works on linux or macos (hopefully it does)
- System prompts are not optimized properly yet to be honest, some bugs regarding phone calls are also present, saying that, it should still "function" relatively well, but let me know if any issues are encountered.
- Sometimes TTS takes a pretty long time, for most of the cases, it seemed to be a cartesian issue but let me know if it happens frequently or not.
- The voice agent is a bit verbose yes can be adjusted with a better sys prompt.
