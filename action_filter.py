from actions import BrowserState


def get_valid_actions(state: BrowserState) -> set[str]:
    if state.in_textbox:
        return {
            "click out",
            "enter text",
            "press enter",
            "press backspace",
            "press delete",
            "cursor left",
            "cursor right",
            "cursor up",
            "cursor down",
            "select all",
            "move line start",
            "move line end",
            "move word left",
            "move word right",
            "hold shift",
            "release shift",
        }
    if state.auto_scroll == "up":
        return {
            "stop scrolling up",
            "continue scrolling up",
            "start scrolling down",
        }
    if state.auto_scroll == "down":
        return {
            "stop scrolling down",
            "continue scrolling down",
            "start scrolling up",
        }

    return {
        "new tab",
        "search action",
        "search url action",
        "scroll up",
        "scroll down",
        "start scrolling up",
        "start scrolling down",
        "click button *",
        "select tab *",
        "close tab *",
    }
