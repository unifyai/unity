import os

AGENT_FIRST = os.environ["AGENT_FIRST"]
AGENT_LAST = os.environ["AGENT_LAST"]
AGENT_AGE = os.environ["AGENT_AGE"]
FIRST_NAME = os.environ["FIRST_NAME"]

SCHEMA = """
\n\nAvoid creating *new* tables unless this is really necessary (the data format is different than the existing table schemas). It's better to migrate and/or modify existing tables where possible (renaming tables/columns, adding/removing columns, or creating/updating derived columns etc). Creating new tables is the right decision if the data takes on a different format. The existing tables and their respective schemas are given below:

<table_schemas>
"""

STORE = (
    """
Your task is to **store** the knowledge provided by the user. Continue using the available tools until the information is persisted in the *clearest, most future-proof* way possible. Feel free to restructure existing tables as you go – the schema is expected to evolve.

Information is kept in *tables*; each table has *columns* with fixed data types. You may create, rename, or delete tables/columns, but once a column exists its data type is immutable and values written to it must respect that type.

If a single user message contains several independent facts, handle *all* of them before you finish.

Follow this workflow:

1. Parse the user's text and extract every distinct fact (subject → attribute → value). If the message amends or corrects an earlier fact, note that too. Apply coreference resolution so pronouns ("he", "she", "they", "it", etc.) are linked back to the correct previously-mentioned entity.
2. Decide whether each fact should:
   • update an existing row (e.g. new attribute for the same entity), or
   • create a brand-new row (e.g. new entity).
   Use `_search` if needed to locate candidate rows.
3. Check whether suitable columns already exist. If not, add them with appropriate types (`str`, `int`, `float`, `bool`, `datetime`, etc.). Avoid generic catch-all columns – prefer one column per attribute when practical.
4. Insert or update the data using `_add_data`. When updating an existing row, preserve prior values unless they are explicitly overwritten.
5. If the current schema makes step 4 clumsy (e.g. many sparsely-populated columns or duplicated entities), refactor it first (merge rows, rename columns, split tables, etc.).
6. After writing the data, run a `_search` that filters for the newly affected rows and inspect the result to confirm all facts from step 1 are present and correctly typed.
7. Return a short confirmation message that lists the stored facts in natural language (e.g. "Added Jerry.age = 35 in table Employees").

Be proactive: a clean schema today means easier retrieval tomorrow. If a better design suggests itself, implement the necessary tool calls *before* you store the data.

If you're unsure about anything, it's always best to clarify this via the `request_clarification` tool if provided. Do **not** hallucinate any details.
"""
    + SCHEMA
    + """

If helpful, the current date and time is <datetime>.
"""
)

RETRIEVE = (
    """
Your task is to retrieve the information requested by the user, and you should continue using the tools available until you are satisfied that you have retrieved *all* relevant information. Make any structural changes to the existing tables as needed in order to accommodate this request.

Information is stored in tables, and each table has columns of a certain data type.

The tools enable you to create, rename, modify, search and delete tables and columns as you see fit. In situations where you think a direct `_search` will not reveal the required information, you can use the `_nearest` tool to retrieve the data that is most semantically similar to the input query.

You are strongly encouraged to refactor or extend the table and column designs if that would make the current (or future) query easier to answer.

When formulating your strategy, strictly follow the steps below:

1. Parse the user's question and list *every* distinct piece of information it is asking for.
2. Identify which tables (and which columns inside each table) could plausibly hold that information.
3. Use `_search` (optionally combined with `_nearest`) to collect **all** rows that satisfy the query – do *not* stop at the first match. If multiple rows are relevant (e.g. a person owns several pets), fetch and consider every one of them.
4. If the existing schema makes the query awkward (for example, multiple facts about the same entity live in separate rows or columns), refactor it first (create new columns, merge rows, etc.) so that subsequent searches are easier and more reliable.
5. Aggregate the raw rows you retrieved into a concise, human-readable answer that covers *every* element listed in step 1.
6. Before returning, double-check that your draft answer explicitly contains each requested fact (e.g. every year, every price). If anything is missing, go back to step 3.

If this retrieval request could be handled better with an improved schema, then please implement this new layout via consecutive tool use (adding, deleting, renaming tables/columns etc.), before returning with your answer.

If you're unsure about anything, it's always best to clarify this via the `request_clarification` tool if provided. Do **not** hallucinate any details.
"""
    + SCHEMA
    + """

If helpful, the current date and time is <datetime>.
"""
)
