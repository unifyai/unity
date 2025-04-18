from dotenv import load_dotenv

load_dotenv()

from primitive import primitive


@primitive
def _open_gmail():
    """Open gmail.com"""


# @observe
# ToDo: implement observe decorator
def _signed_in():
    """Check if user is currently signed into Gmail."""


@primitive
def _click_sign_in():
    """Click the sign in button."""


@primitive
def _click_username_field():
    """Click the username input field."""


@primitive
def _enter_username():
    """Enter the Gmail username."""


@primitive
def _click_password_field():
    """Click the password input field."""


@primitive
def _enter_password():
    """Enter the Gmail password."""


@primitive
def _click_enter():
    """Press the Enter key."""


def _defer_to_llm():
    raise Exception("Not yet implemented")


def log_into_gmail():
    """Log into GMail using known credentials."""
    _open_gmail()
    # if _signed_in():
    #     return "Already signed in"
    _click_sign_in()
    _click_username_field()
    _enter_username()
    _click_password_field()
    _enter_password()
    _click_enter()
    # if not _signed_in():
    #     _defer_to_llm()
    return "Signed in successfully"
