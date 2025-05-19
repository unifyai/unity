import sys
import functools
import traceback
import unify


def _handle_project(test_fn=None, try_reuse_prev: bool = False, delete_on_exit=False):

    if test_fn is None:
        return lambda f: _handle_project(
            f,
            try_reuse_prev=try_reuse_prev,
            delete_on_exit=delete_on_exit,
        )

    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        project = test_fn.__name__
        if not try_reuse_prev and project in unify.list_projects():
            unify.delete_project(project)
        try:
            with unify.Project(project):
                unify.traced(test_fn)(*args, **kwargs)
            if delete_on_exit:
                unify.delete_project(project)
        except:
            if delete_on_exit:
                unify.delete_project(project)
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_string = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            raise Exception(f"{tb_string}")

    return wrapper
