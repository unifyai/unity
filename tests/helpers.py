import sys
import functools
import traceback
import unify


def _handle_project(test_fn=None, delete_on_cleanup=False):

    if test_fn is None:
        return lambda f: _handle_project(
            f,
            delete_on_cleanup=delete_on_cleanup,
        )

    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        project = test_fn.__name__
        if project in unify.list_projects():
            unify.delete_project(project)
        try:
            with unify.Project(project):
                unify.traced(test_fn(*args, **kwargs))
            if delete_on_cleanup:
                unify.delete_project(project)
        except:
            if delete_on_cleanup:
                unify.delete_project(project)
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_string = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            raise Exception(f"{tb_string}")

    return wrapper
