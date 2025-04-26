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

Listens to the Transcript Queue, and for every new segment that arrives, the manager checks if the new segment is requesting a new task to be triggered or an existing task to be modified. This is combined with the prior context of the conversation as well. Optionally the manager can also have access to the `TranscriptManager` if the fixed context window provided isn't enough for full clarity (ie "Could you start working on the task I mentioned last week?"). For every segment of dialogue which **is** deemed to represent a task-related request, the manager parses the dialogue and extracts a clean and clearly written request, updates the flat set of tasks (name + description) stored on the backend (if needed), potentially including pending tasks and scheduled tasks. Then, if (a) a new task is requested to start immediately or (b) the change pertains to a task currently underway, then also publuish the task request on the Task Queue (for the active `Planner` to receive).

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
