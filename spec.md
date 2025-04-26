# Unity Technical Specification


## Queues

### Transcript Queue
The latest user-agent transcript segment (since the last user message). `ComsManager` publishes after **every new user message**, and `TaskManager` subscribes.

### Task Queue
User task trigger and task update requests, in text form. `TaskManager` publishes, and `Planner` subscribes.

### Action Queue
Low-level browser actions in text form. `Planner` publishes, and `Controller` subscribes.

### Browser Command Queue
Lower-level browser commands, based on the exact set of commands defined in `controller/commands.py`. `Controller` publishes, and `BrowserWorker` subscribes.

### Browser State Queue
The state of the browser (y axis scroll in pixels , "in text-box" bool etc). `BrowserWorker` publishes, `Controller` subscribes.

### Action Completion Queue
Actions which have been completed, referenced by their title (plain text). `Controller` publishes, `Planner` subscribes.

### Task Completion Queue
Tasks which have been completed, referenced by their title (plan text). `Planner` publishes, `TaskManager` and `ComsManager` both subscribe.


## ComsManager

Manages all text-based and voice-based communcation, whether it be with the user, another person, or another AI assistant. When you call your assistant, it is the `ComsManager` that picks up the phone, when you send a text, the `ComsManager` parses it and decided whether or not to respond.

### Tools

#### Probe Transcript

Can send text-based queries to the `TranscriptManager.ask`, asking anything related to the prior communications.

#### Probe Knowledge

Can send text-based queries to the `KnowledgeManager.ask`, asking anything related to useful knowledge the assistant has accumulated.

### Publishes To

#### Transcript Queue

Every time the user sends a message or ends their turn during a call, the `transcript_q` is populated with all messages *since the last user message*.

### Subscribed To

#### Task Completion Queue

Every time a task is completed, the `ComsManager` is informed, enabling this to be expressed to the user if appropriate.


## TranscriptManager

Manages all searches across the entire set of transcripts with other users, across all commumication mediums. Receives text-based questions and uses the available tools to search the backend which stores the entire set of transcripts.

### Tools

#### Get Summaries

Can get exchange summaries flexibly filtered by the sender, receiver, timestamps, summary content, summary length etc. (uses `get_logs` filtering)

#### Get Messages

Can get messages flexibly filtered by the sender, receiver, timestamps, message content, message length etc. (uses `get_logs` filtering)

### Called By

#### ComsManager

The `ComsManager` is able to ask general questions, which the `TranscriptManager` must then try to answer as well as possible using the available tools.


## KnowledgeManager

Manages all searches across the entire set of stored knowledge with other users, across all commumication mediums. Receives text-based questions and uses the available tools to search the backend which stores the entire set of transcripts. Also *automatically* triggered at the *end* of *every* exchange, with a general task to "Update the stored knowledge with any new and important information based on the latest conversation, if relevant and useful. Feel free to restructure existing knowledge representation if a new format is now more appealing."

### Tools

TBD - A pretty expressive set of tools (built of top of our logging backend), exposing the full power of multiple tables (contexts), arbitrary columns, different data types etc., depending on the knowledge which must be stored.

### Called By

#### ComsManager

The `ComsManager` is able to ask general questions, which the `KnowledgeManager` must then try to answer as well as possible using the available tools.


## TaskManager

Listens to the Transcript Queue, and for every new segment that arrives, the manager checks if the new segment is requesting a new task to be triggered or an existing task to be modified. This is combined with the prior context of the conversation as well. The manager also has access to the `TranscriptManager`, which it can use if the fixed context window provided isn't enough for full clarity (ie "Could you start working on the task I mentioned last week?"). For every segment of dialogue which **is** deemed to represent a task-related request, the manager parses the dialogue and extracts a clean and clearly written request, sends the update to the TaskListManager to update the flat set of tasks (name + description) stored on the user account (if needed), potentially including pending tasks and scheduled tasks. Then, if (a) a *new* task is requested to start *immediately* or (b) a *change* is requested for a task *currently* underway, then also publuish the task request on the Task Queue (for the active `Planner` to receive, and either trigger a new task or update a live one).

### Tools

#### Probe Transcript

The `TaskManager` can send text-based queries to the `TranscriptManager.ask`, asking anything related to the prior communications, in rare cases where the immediate context isn't enough to determine the task, such as "Could you start working on the task I mentioned last week?".

#### Probe Task List

If changes are requested in the transcript, then the manager can send text-based queries to the `TaskListManager.ask`, asking anything related to the set of active, completed, pending and scheduled tasks.

#### Update Task List

If changes are requested in the transcript, then the manager can send text-based commands to the `TaskListManager.update`, asking the manager to update the task list in a specific manner.

### Subscribed To

#### Transcript Queue

Every time the user sends a message or ends their turn during a call, the `transcript_q` is populated with all messages *since the last user message*. It is down to the task manager to accumulate these and create windowed context, if needed.

### Publishes To

#### Task Queue

If (a) a *new task* is requested to start *now* OR (b) *changes* are required for an *active* task, then the text-based task update is published on the task queue, for the planner to either (a) initiate the new task or (b) modify the active task.


## TaskListManager

Manages all searches and updates across the list of tasks stored in the user account, for *this* assistant. Receives text-based questions and update requests, and uses the available tools to search and update the task list stored in the backend.

### Tools

#### Get Tasks

Can get tasks flexibly filtered by the `assigned_at` and `last_performed` timestamps, `status` flag (active, completed, in progress etc.), recurring frequency, description length, contains or does not contain sub-string etc. (uses `get_logs` filtering)

#### Delete Task

Removes the task for the task list.

#### Change Task Status

Change the status of a task

#### Change Task Description

Change the description of a task.

#### Reschedule Task

Change the scheduled time for a task

#### Merge Tasks

Combine multiple tasks into a single task

### Called By

#### TaskManager

The `TaskManager` is able to ask general questions and submit general requests, both in plain text form, which the `TaskListManager` must then answer and/or perform for the task manager.


## Planner

Receives a stream of user inputs related to this task (or a newly created task at the beginning). The requests can either be high-level or low-level guidance, fully text-based in either case. The planner must then stream a series of low-level actions to the `Controller`, as quickly and efficiently as possible in order to complete the task (so the task is not delayed, especially important in cases with live agent-user interaction).

### Subscribed To

#### Task Queue

Every time a task update (or task trigger if there are no active tasks) is made, then this is placed on the queue, and the planner will receice it.

### Publishes To

#### Action Queue

Send sequential text-based actions, each of which corresponds to a *single* action, which would correspond to a single browser command (see `controller/commnds.py`)


## Controller

The controller listens for text-based single actions to complete, listens for the latest browser state, uses both to convert the latest action into a well defined browser command (`controller/commands.py`), and then sends this command to the `BrowserWorker` for execution in the browser. It then informs the `Planner` (up the stack) when the action has been completed.

### Subscribed To

#### Action Queue

Listens for text-based actions to complete, sent by the `Planner`.

#### Browser State Queue

Listens for the latest browser state (y axis scroll [pixels], in-textbox [bool], current-url [str], all tabs List[str], all clickable buttons List[str]  etc.), and uses this alongside action description to select an appropriate action browser action (`controller/commands.py`).

### Publishes To

#### Browser Command Queue

Sends the well-defined fixed browser commands (`controller/commands.py`) to the `BrowserWorker`.

#### Action Completion Queue

Once an action is completed, send the text-based action received back to this queue, to inform the `Planner`.
