import sys
import functools
import traceback
import unify


def _handle_project(
    test_fn=None,
    try_reuse_prev_ctx: bool = False,
    delete_ctx_on_exit=False,
):

    if test_fn is None:
        return lambda f: _handle_project(
            f,
            try_reuse_prev_ctx=try_reuse_prev_ctx,
            delete_ctx_on_exit=delete_ctx_on_exit,
        )

    @functools.wraps(test_fn)
    def wrapper(*args, **kwargs):
        ctx = test_fn.__name__
        if not try_reuse_prev_ctx and ctx in unify.get_contexts():
            unify.delete_context(ctx)
        try:
            with unify.Context(ctx):
                unify.traced(test_fn)(*args, **kwargs)
            if delete_ctx_on_exit:
                unify.delete_context(ctx)
        except:
            if delete_ctx_on_exit:
                unify.delete_context(ctx)
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_string = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            raise Exception(f"{tb_string}")

    return wrapper
