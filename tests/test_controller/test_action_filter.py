from unity.controller import action_filter as af
from unity.controller.states import BrowserState
from unity.controller import commands as cmd


def test_in_textbox_includes_textbox_commands():
    st = BrowserState(in_textbox=True)
    valid = af.get_valid_actions(st, mode="actions")
    # Every textbox command should be included
    assert cmd.TEXTBOX_COMMANDS <= valid
    # still allows scrolling
    assert cmd.CMD_SCROLL_DOWN in valid


def test_dialog_prompt_limits_actions():
    st = BrowserState(dialog_open=True, dialog_type="prompt")
    valid = af.get_valid_actions(st, mode="actions")
    assert {cmd.CMD_ACCEPT_DIALOG, cmd.CMD_DISMISS_DIALOG, cmd.CMD_TYPE_DIALOG} <= valid
    # normal navigation should be blocked
    assert cmd.CMD_SCROLL_DOWN not in valid
    assert cmd.CMD_OPEN_URL not in valid


def test_captcha_pending_restricts_actions():
    st = BrowserState(captcha_pending=True)
    valid = af.get_valid_actions(st, mode="actions")
    assert valid == {cmd.CMD_SCROLL_UP, cmd.CMD_SCROLL_DOWN, cmd.CMD_STOP_SCROLLING} 