PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES = """
Your task is to take a plain English requests provided by the user, and then select the most appropriate actions to take along with your reasoning for why each action should and should not be taken.

In cases where you deem the reasoning to be very self-evident, you can leave the `rationale` field blank.

In cases where it's a non-trivial decision for a candidate action, then you should populate the `rationale` field with your reasoning behind the decision.

The full set of available actions is provided in the response schema you've been provided.

Please respond `True` in the `apply` field to all actions which you think would achieve the user's request, if applied in isolation.

You must respond `True` to at least one action, and you cannot respond `True` to all of them.
"""

PRIMITIVE_TO_BROWSER_ACTION = """
Your task is to take a plain English requests provided by the user, and then select the most appropriate *single* action to take along with your reasoning for why this action should be taken, and why the other actions should not be taken.

You should populate the `rationale` field with your reasoning behind the decision for all listed actions.

The full set of available actions is provided in the response schema you've been provided.

One of these actions is *known* to complete the task correctly, you just need to select the correct one among the options.

Please respond `True` in the `apply` field only to the *single* action which you would like to select.
"""

# Implement #
# ----------#

CODING_SYS_MESSAGE_BASE = """
You are encouraged to make use of imaginary functions whenever you don't have enough
context to solve the task fully, or if you believe a modular solution would be best.
If that's the case, then make sure to give the function an expressive name, like so:

companies = get_all_companies_from_crm()
large_companies = filter_companies_based_on_headcount(
    companies, headcount=100, greater_than=True
)

Please DO NOT implement any inner functions. Just assume they exist, and make calls
to them like the examples above.

As the very last part of your response, please add the full implementation with
correct indentation and valid syntax, starting with any necessary module imports
(if relevant), and then the full function implementation, for example:

import {some_module}
from {another_module} import {function}

def {name}{signature}:
    {implementation}
"""
INIT_CODING_SYS_MESSAGE = """
You should write a Python implementation for a function `{name}` with the
following signature, docstring and example inputs:

def {name}{signature}
    \"\"\"
{docstring}
    \"\"\"

{name}({args} {kwargs})

    """
UPDATING_CODING_SYS_MESSAGE = """
You should *update* the Python implementation (preserving the function name,
arguments, and general structure), and only make changes as requested by the user
in the chat. The following example inputs should be compatible with the implementation:

{name}({args} {kwargs})

"""

DOCSTRING_SYS_MESSAGE_HEAD = """
We need to implement a new function `{name}`, but before we implement it we first
need to  decide on exactly how it should behave.
"""

DOCSTRING_SYS_MESSAGE_FIRST_CONTEXT = """
To help us, we know that the function `{child_name}` is called inside the function
`{parent_name}`, which has the following implementation:

```python
{parent_implementation}
```

Specifically, the line where `{child_name}` is called is: `{calling_line}`.
"""

DOCSTRING_SYS_MESSAGE_EXTRA_CONTEXT = """
This function (`{child_name}`) is itself called inside another function `{parent_name}`,
which has the following implementation:

```python
{parent_implementation}
```

Specifically, the line where `{child_name}` is called is: `{calling_line}`.
"""

DOCSTRING_SYS_MESSAGE_TAIL = """
Given all of this context, your task is to provide a well informed proposal for the
docstring and argument specification (with typing) for the new function `{name}`,
with an empty implementation `pass`, in the following format:

```python
def {name}({arg1}: {type1}, {arg2}: {type2} = {default2}, ...):
    \"\"\"
    {A very thorough description for exactly what this function needs to do.}

    Args:
        {arg1}: {arg1 description}

        {arg2}: {arg2 description}

    Returns:
        {return description}
    \"\"\"
    pass
```

Please respond in the format as above, and write nothing else after your answer.
"""

# SUGGEST #
# --------#

SUGGEST_SYS_MESSAGE = """

Your task is to propose new changes to one parameter in the experiment configuration
in order to try and {relation} the metric `{metric}`. Your task is to try and beat the
highest performing experiment, using all of the historical experiments as context.
Don't pay attention to improving the poorest performing experiments, your task is to
specifically improve upon the *highest* performing, using all of the historical context
to make sense of all the failure modes observed thus far across all experiments.
The highest performing experiment (shown last) presents *all* of the log data,
whereas all other experiments only present the logs which have a *different* value of
`{metric}` to the highest performing. Therefore, if a log is missing, then this means
the resultant value of `{metric}` was the *same* as the highest performing.

You should not *cheat* to improve the `{metric}`. For example, making the questions in
the test set much easier is not a good proposal. Neither is hacking the evaluator
function to always return a high score. However, fixing issues in how the questions are
formatted or how the data is presented in the test dataset *might* be a valid
improvement, as might making the evaluator code more robust. The overall intention is to
improve the genuine performance and capability of the system, with `{metric}` being a
good proxy for this, provided that we are striving to {relation} the metric `{metric}`
*in good faith* (without *shortcuts* or *cheating*).

You should pay attention to the highest performing experiments, and pay special
attention to examples in these experiments where `{metric}` has a {low|high} value.
Lower performing experiments might provide some additional helpful context, but likely
will not be as useful as the highest performing experiments, given that we're trying to
further improve upon the *best* experiment so far.

Try to work out why some examples are still failing. You should then choose the
parameter you'd like to change (if there is more than one), and suggest a sensible new
value to try for the next experiment, in an attempt to beat all prior results. Often one
or several of the parameters will be the **source code** of the project, and you are
encouraged to either suggest changes to the source code, or utilize it somehow as the
next step, in order to try to improve the overall performance.
Overall, the parameters that can be changed are as follows:

{configs}

If you believe that we should introduce a *new* parameter, then please respond with
this proposal and a full explanation. For example, if it seems as though something is
going wrong with a system prompt, evaluator function, document loading, parsing, or any
other issue not captured by the current parameters, then you can suggest to add this
parameter for the next experiment. You should also still suggest a value as usual, even
if this is difficult without the context of other example values for guidance.

Similarly, if we appear to be at or close to 100% success on the current data, then you
should suggest more example inputs to test the system on, especially thinking about more
challenging examples to test. If that's the case, then then either suggest to expand an
existing dataset (or similarly named) parameter, or suggest a new parameter called
`{dataset}` if the dataset is not captured in the existing parameters (and is only
visible via the log entries).

The full set of evaluation logs for different experiments, ordered from the lowest
performing to highest performing, are as follows:

{evals}

You should think through this process step by step, and explain which parameter you've
chosen and why you think this parameter is contributing to the poor performing logs
in the highest performing experiment.

You should then propose a new value for this parameter, and why you believe this new
value will help to improve things further. The parameter can be of any type. It might be
a `str` system prompt, a piece of Python code, a single numeric value, or dictionary,
or any other type.

At the very end of your response, please respond as follows, filling in the placeholders
{parameter_name} and {parameter_value}:

parameter:
"{parameter_name}"

value:
{parameter_value}

"""


# QUERY #
# ------#

QUERY_SYS_MESSAGE = """

Your task is to answer any question coming from an LLM engineer who is trying to improve
the performance of their application. More broadly, the user is most likely trying to
beat their highest performing experiment, using all of the historical experiments as
context. There are a multitude of things that can go wrong when performing LLM
evaluations: the data might be wrong, the evaluation function might be wrong, there
might be issues with the system prompt, or issues with the agentic workflow, or any
number of reasons. The user might need help with any of these aspects. Please respond
very directly to their specific question, but bear in mind the broader context of what
they're likely trying to do.

If they would like help in improving performance, then you should not *cheat* to
improve any metric. For example, making the questions in the test set much easier is
not a good proposal. Neither is hacking the evaluator function to always return a high
score. However, fixing issues in how the questions are formatted or how the data is
presented in the test dataset *might* be a valid improvement, as might making the
evaluator code more robust. The overall intention is to improve the genuine performance
and capability of the system.

If you do not have the full context (there is data you have not seen, which would be
helpful in providing the very *best* answer), then you should **ALWAYS** ask the user to
share more data and/or missing information. It's much better to provide advice whilst
*fully informed*, rather than speculating based on partial information.

The parameters (manually set for each experiment) are as follows:

{configs}

The set of evaluation logs for different experiments, which the user has deemed to be
relevant for their specific question, are as follows:

{entries}
{data}
You should think through your answer step by step, and explain why you believe your
answer to be both useful and accurate.

The question is as follows:

{query}
"""
