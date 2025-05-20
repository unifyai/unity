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
        file_path = test_fn.__code__.co_filename
        test_path = "/".join(file_path.split("/tests/")[1].split("/"))[:-3]
        ctx = f"{test_path}/{test_fn.__name__}" if test_path else test_fn.__name__

        if not try_reuse_prev_ctx and unify.get_contexts(prefix=ctx):
            unify.delete_context(ctx)
        try:
            with unify.Context(ctx):
                unify.set_trace_context("Traces")
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
