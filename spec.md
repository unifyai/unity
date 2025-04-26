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

Manages all text-based and voice-based communcation, whether it be with the user, another person, or another AI assistant. When you call your assistant, it is the ComsManager that picks up the phone, when you send a text, the coms manager parses it and decided whether or not to respond.

### Tools

#### Probe Transcript

The `ComsManager` can send text-based queries to the `TranscriptManager.ask`, asking anything related to the prior communications.

#### Probe Knowledge

The `ComsManager` can send text-based queries to the `KnowledgeManager.ask`, asking anything related to useful knowledge the assistant has accumulated.

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
