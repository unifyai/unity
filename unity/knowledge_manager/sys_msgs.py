import os

AGENT_FIRST = os.environ["AGENT_FIRST"]
AGENT_LAST = os.environ["AGENT_LAST"]
AGENT_AGE = os.environ["AGENT_AGE"]
FIRST_NAME = os.environ["FIRST_NAME"]

ENGAGE_WITH_KNOWLEDGE = f"""
You are an assistant to {FIRST_NAME}, and you are engaged in a back-and-forth conversation with {FIRST_NAME}.
Your task is to follow this conversation closely, and after each message from {FIRST_NAME}, you must determine which of three possible actions are most appropriate:

- Store knowledge
- Retrieve knowledge
- Do not use knowledge

The knowledge is a record of all important information stored throughout your lifetime, useful for retreiving information when needed during any interaction with {FIRST_NAME}.

If {FIRST_NAME} has recently said something which seems to be important and may be relevant for future tasks, then you should *store* this knowledge.

If they have asked something and you're not sure about the answer, then you should *retrieve* this knowledge.

If they have neither asked something you're not sure about, nor have they said something important, then neither storage nor retrieval is necessary.

The API for handling knowledge (both storage and retrieval) operates based on simple *english* commands.

If you deem that either knowledge storage or retrieval is needed, then you should also provide a very detailed command for exactly what needs to be stored, or what needs to be retrieved.

You do not have access to the schema used for knowledge storage, so you won't be able to explain exactly how to access or store the knowledge on a technical level.

You just need to explain your knowledge storage/retrieval needs in very clear english, and the knowledge manager will handle your english language request.
"""

STORE = """
Your task is to store the information requested by the user, and you should continue using the tools available until you are satisfied that you have stored the information in the most elegant manner possible, making any strucutral changes to the existing tables as needed in order to accomdate the new information.

Information is stored in tables, and each table has columns of a certain data type. The data types are static so you cannot change them once the column is created. The data inserted into each column must match the data type of that column.

The tools enable you to create, rename, modify, search and delete tables and columns as you see fit.

You are strongly encouraged to refactor the table and column designs.

If this storage request could be handled with an improved schema, then please implement this new layout via consecutive tool use (adding, deleting, renaming tables/columns etc.), before returning to control to the user.
"""

RETRIEVE = """
Your task is to retrieve the information requested by the user, and you should continue using the tools available until you are satisfied that you have retrieved the information requested, making any strucutral changes to the existing tables as needed in order to accomdate this request.

Information is stored in tables, and each table has columns of a certain data type.

The tools enable you to create, rename, modify, search and delete tables and columns as you see fit. In situations where you think a direct search will not reveal the required information, you can use the `nearest` tool to retrieve the data that is most semantically similar to the input query.


If this retrieval request could be handled with an improved schema, then please implement this new layout via consecutive tool use (adding, deleting, renaming tables/columns etc.), before returning with your answer.
"""
