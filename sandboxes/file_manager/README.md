File Manager Sandbox
====================

This folder contains an **interactive playground** for the `FileManager` component that lives in `unity/file_manager/`.  The goal of the sandbox is to let you experiment with the manager in isolation – import files, parse their contents, manage file operations, and observe how the underlying tool-loop behaves before you integrate the manager into a larger system.

### Video walkthroughs

- Coming soon...

What is the `FileManager`?
--------------------------
`FileManager` is a **powerful read-only** file analysis system that manages files received or downloaded during a Unity session. Files are automatically added when:

- **Email attachments** are received
- **Browser downloads** are completed
- **Other system integrations** add files

It supports multiple file formats (`.txt`, `.pdf`, `.docx`) and provides sophisticated content analysis capabilities:

### Core Methods
* **`ask(filename, question)`** – Ask specific questions about individual file contents
* **`search_files(references, k)`** – Semantic search across all files by content similarity
* **`filter_files(filter)`** – Exact filtering by criteria like filename, status, metadata
* **`parse(filename)`** – Extract structured data from files
* **`list()`** – List all available files
* **`exists(filename)`** – Check if a file is available
* **`import_file(path)`** – Import a single file from filesystem
* **`import_directory(path)`** – Import all files from a directory

### Advanced Capabilities
The FileManager automatically parses imported files and stores structured content in a cloud-based table, enabling:

- **Semantic Search**: Find files by content similarity using natural language queries
- **Cross-Document Analysis**: Compare and analyze content across multiple files
- **Metadata Filtering**: Filter by file properties, processing status, size, type, etc.
- **Persistent Storage**: Parsed content is stored and searchable without re-parsing

Under the hood, `ask` launches a _tool-loop_ where an LLM can call comprehensive read-only tools (`parse`, `search_files`, `filter_files`, `list`, `exists`) until it reaches a final answer. The extensive unit-test suite in `tests/test_file_manager/` exercises all functionality – including ranking precision tests that verify semantic search returns the most relevant documents first.

### Features

- **Semantic Search**: Find files by content similarity using embeddings and cosine distance ranking
- **Advanced Filtering**: Exact criteria filtering with support for complex expressions
- **Multi-format parsing**: Supports text files, PDFs, and Word documents using the integrated DoclingParser
- **Persistent Storage**: Automatically logs parsed files to cloud table for fast retrieval
- **Cross-document analysis**: Compare content, extract information across multiple files
- **Content intelligence**: Query file contents, extract key information, and perform document analysis
- **Read-only operations**: Safe content queries without file modification
- **Voice interaction**: Full voice mode support with speech-to-text and text-to-speech
- **Steerable tools**: In-flight steering to interject and control runs without restarting
- **Clarification requests**: Interactive clarification flows for ambiguous questions

Running the sandbox
-------------------
The entry-point lives at `sandboxes/file_manager/sandbox.py` and can be executed directly or via Python's `-m` switch:

```bash
# Basic text-only session
python -m sandboxes.file_manager.sandbox

# The same, but enable voice I/O via Deepgram + Cartesia
python -m sandboxes.file_manager.sandbox --voice
```

CLI flags
~~~~~~~~~
`sandbox.py` re-uses the common helper in `sandboxes/utils.py`, so it shares a standard set of options:

```
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d        Show full reasoning steps of every tool-loop
--traced / -t       Wrap manager calls with unify.traced for detailed logs
--project_name / -p Name of the Unify **project/context** (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
--no_clarifications Disable clarification requests (auto-proceed with best guesses)
--log_level         Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)
--log_file          Optional file path for logs (default: console only)
--log_tcp           Enable TCP log streaming on localhost:9999
--show_requests     Stream all Unify API requests to the terminal
```

The defaults work well for getting started, but `--voice` is especially fun once you have Deepgram and Cartesia API keys in your environment.

Getting started
---------------
When you first run the sandbox, you'll see a command prompt that accepts both structured commands and free-form natural language. Here are some ways to get started:

### Quick commands

```
us "populate sample files"        # Load sample files for testing
lf                                # List all available files
pf document.pdf                   # Parse and analyze a specific file
search AI machine learning        # Search files by content similarity
filter filename.endswith('.pdf')  # Filter files by exact criteria
lc                                # List table columns/schema
if /path/to/file.pdf              # Import a single file
id /path/to/directory             # Import all files from directory
```

### Natural language examples

```
"What are the main topics discussed in the quarterly report?"
"Find all files that mention artificial intelligence"
"Which documents contain security policies?"
"Compare the technical specifications across all PDF files"
"Search for files about machine learning algorithms"
"What is the difference between report1.pdf and report2.pdf?"
"Extract all the compliance requirements from policy documents"
"Find documents that discuss IoT devices and connectivity"
```

### Advanced search examples

```
search "IT security policy compliance GDPR"     # Semantic content search
filter "status == 'success' and metadata['file_size'] > 1000000"  # Complex filtering
search "neural networks deep learning"          # Find AI/ML documents
filter "filename.endswith('.docx')"            # Find Word documents
```

### Scenario seeding

Use `us` or `usv` (voice mode) to populate the file manager with sample files for testing:

```
us "Demonstrate semantic search across IT policy and technical documentation"
us "Showcase cross-document analysis and filtering capabilities"
us "Load sample files and show content intelligence features"
usv  # Voice-guided scenario creation
```

The sandbox automatically pre-loads sample files (IT Policy Document and SmartHome Hub Documentation) to demonstrate:
- Semantic search across different document types
- Domain-specific content filtering
- Cross-document analysis and comparison
- Content extraction and intelligence workflows

### Voice mode

With `--voice`, you can:
- **Speak commands**: Press Enter on an empty line to start voice recording
- **Voice steering**: Say "pause", "stop", or "cancel" during LLM operations
- **TTS playback**: Hear responses read aloud (press Enter to skip)
- **Voice clarifications**: Respond to clarification requests by voice

Architecture notes
------------------
The sandbox demonstrates several key patterns:

1. **Read-only design**: All operations are safe content queries without file modification
2. **Semantic search architecture**: Embeddings-based search with cosine distance ranking
3. **Persistent storage**: Automatically logs parsed files to cloud table for instant retrieval
4. **Multi-column references**: Search across content, metadata, and derived expressions
5. **File format detection**: Dynamically determines supported formats and parsing strategies
6. **Content extraction**: Leverages DoclingParser for robust document parsing
7. **Tool composition**: Shows how search, filter, parse, and analysis tools work together
8. **Error handling**: Graceful handling of missing files and parsing failures

The `FileManager` maintains both a local registry (display names → file paths) and a cloud-based table (parsed content + metadata) that enables powerful semantic search and filtering. The architecture mirrors ContactManager patterns while maintaining read-only semantics. Files are automatically parsed and logged when imported, creating a searchable knowledge base.

File formats & parsing
----------------------
Currently supported formats:
- **Text files** (`.txt`): Direct content extraction with encoding detection
- **PDF documents** (`.pdf`): Layout-aware parsing with text and structure extraction
- **Word documents** (`.docx`): Native Office document parsing with formatting preservation

The parsing system is extensible – new formats can be added by implementing the `BaseParser` interface.

Development notes
-----------------
For FileManager development:
- The core implementation lives in `unity/file_manager/`
- Tests are in `tests/test_file_manager/` including comprehensive search ranking tests
- Search/filter functionality tests in `tests/test_file_manager/test_file_manager_search_filter.py`
- Parser-specific tests are in `tests/test_file_manager/test_parser/`
- The sandbox provides an isolated environment for testing new features

This sandbox is particularly useful for:
- **Testing semantic search precision**: Verify ranking and relevance with k=1 queries
- **Cross-document analysis**: Compare content across multiple file types
- **Advanced filtering workflows**: Test complex filtering expressions and metadata queries
- **Content intelligence**: Extract structured information from unstructured documents
- **Search ranking validation**: Ensure most relevant documents appear first
- **Multi-column search**: Test search across content, metadata, and derived fields
- Validating parsing behavior across different file formats
- Experimenting with natural language content queries
- Debugging document parsing and extraction logic

### Key Testing Scenarios
The sandbox includes sample files (IT Policy + SmartHome Documentation) specifically chosen to test:
- Domain-specific search precision (IT vs IoT content)
- Technical specification extraction
- Policy and compliance information retrieval
- Cross-document comparison and analysis
