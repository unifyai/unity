# FileManager & GlobalFileManager Sandboxes

Interactive command-line environments for testing and exploring the **FileManager** and **GlobalFileManager** components of the Unity framework.

---

## Overview

This directory contains two interactive sandboxes:

1. **FileManager Sandbox** (`file_manager_sandbox.py`) - For testing a single filesystem adapter
2. **GlobalFileManager Sandbox** (`global_file_manager_sandbox.py`) - For testing operations across multiple filesystems

Both sandboxes provide full access to the managers' public API methods (`ask`, `ask_about_file`, `organize`) with support for:
- Voice input/output (optional)
- Clarification requests during execution
- Mid-flight steering (pause/interject/cancel)
- Multiple filesystem adapters (Local, CodeSandbox, Interact, Google Drive)
- Scenario-based testing with synthetic file generation

---

## FileManager Sandbox

### Purpose

Test and explore a single FileManager instance with your choice of filesystem adapter. Supports file operations, content analysis, semantic search, and organization tasks.

### Usage

```bash
# Basic usage with default local adapter
python -m sandboxes.file_manager.file_manager_sandbox

# With specific adapter
python -m sandboxes.file_manager.file_manager_sandbox --adapter local --root /tmp/my_files

# With voice mode
python -m sandboxes.file_manager.file_manager_sandbox --voice

# With tracing enabled
python -m sandboxes.file_manager.file_manager_sandbox --traced

# With custom project
python -m sandboxes.file_manager.file_manager_sandbox --project_name MyProject --overwrite
```

### Command Line Arguments

| Argument | Alias | Default | Description |
|----------|-------|---------|-------------|
| `--adapter` | `-a` | `local` | Filesystem adapter: `local`, `codesandbox`, `interact`, `google_drive` |
| `--root` | `-r` | `None` | Root directory for local adapter (creates temp dir if not specified) |
| `--voice` | `-v` | `False` | Enable voice input/output |
| `--traced` | `-t` | `False` | Enable Unify tracing |
| `--debug` | `-d` | `False` | Show reasoning steps |
| `--project_name` | `-p` | `Sandbox` | Unify project name |
| `--overwrite` | `-o` | `False` | Overwrite existing project data |
| `--no_clarifications` | | `False` | Disable clarification requests |
| `--log_in_terminal` | | `False` | Print logs to terminal |
| `--log_tcp_port` | | `None` | TCP port for log streaming |
| `--http_log_tcp_port` | | `None` | TCP port for HTTP log streaming |
| `--project_version` | | `-1` | Project version to load (-1 for latest) |

### Interactive Commands

Once in the sandbox, you can use these commands:

| Command | Description |
|---------|-------------|
| `us {description}` | Generate synthetic file scenario from text description |
| `usv` | Generate synthetic file scenario from voice (requires `--voice`) |
| `save_project` or `sp` | Save current project state |
| `help` or `h` | Show command list |
| `quit` or `exit` | Exit sandbox |
| Any other text | Automatically routed to `ask`, `ask_about_file`, or `organize` |

### Examples

#### Basic File Query
```
command> What files do I have?
[ask] → You currently have 5 files: report.pdf, data.csv, notes.txt, image.png, slides.pptx
```

#### File-Specific Question
```
command> What's the main topic of report.pdf?
[ask_about_file] → The report discusses Q3 sales performance across EMEA regions...
```

#### File Organization
```
command> Rename data.csv to q3_data.csv
[organize] → Successfully renamed 'data.csv' to 'q3_data.csv'
```

#### Scenario Generation
```
command> us Create 10 sample documents about AI research papers
[generate] Building synthetic file scenario – this can take a moment…
📁 Pre-populated 10 sample files: paper1.pdf, paper2.pdf, ...
```

---

## GlobalFileManager Sandbox

### Purpose

Test and explore cross-filesystem operations with multiple filesystem adapters simultaneously. Ideal for testing file synchronization, cross-filesystem search, and multi-source content analysis.

### Usage

```bash
# Basic usage with single local filesystem
python -m sandboxes.file_manager.global_file_manager_sandbox

# With multiple filesystems
python -m sandboxes.file_manager.global_file_manager_sandbox --filesystems local,interact

# With custom local root
python -m sandboxes.file_manager.global_file_manager_sandbox --filesystems local --local-root /tmp/my_files

# With voice mode
python -m sandboxes.file_manager.global_file_manager_sandbox --voice --filesystems local,codesandbox

# With tracing enabled
python -m sandboxes.file_manager.global_file_manager_sandbox --traced --filesystems local,interact,google_drive
```

### Command Line Arguments

| Argument | Alias | Default | Description |
|----------|-------|---------|-------------|
| `--filesystems` | `-f` | `local` | Comma-separated list: `local`, `codesandbox`, `interact`, `google_drive` |
| `--local-root` | | `None` | Root directory for local adapter (creates temp dir if not specified) |
| `--voice` | `-v` | `False` | Enable voice input/output |
| `--traced` | `-t` | `False` | Enable Unify tracing |
| `--debug` | `-d` | `False` | Show reasoning steps |
| `--project_name` | `-p` | `Sandbox` | Unify project name |
| `--overwrite` | `-o` | `False` | Overwrite existing project data |
| `--no_clarifications` | | `False` | Disable clarification requests |
| `--log_in_terminal` | | `False` | Print logs to terminal |
| `--log_tcp_port` | | `None` | TCP port for log streaming |
| `--http_log_tcp_port` | | `None` | TCP port for HTTP log streaming |
| `--project_version` | | `-1` | Project version to load (-1 for latest) |

### Interactive Commands

Once in the sandbox, you can use these commands:

| Command | Description |
|---------|-------------|
| `us {description}` | Generate synthetic file scenario across filesystems |
| `usv` | Generate synthetic file scenario from voice (requires `--voice`) |
| `save_project` or `sp` | Save current project state |
| `help` or `h` | Show command list |
| `quit` or `exit` | Exit sandbox |
| Any other text | Automatically routed to `ask` or `organize` |

### Examples

#### Cross-Filesystem Query
```
command> What files do I have across all filesystems?
[ask] → Found 15 files across 2 filesystems:
  - local: 8 files (reports, data, notes)
  - interact: 7 files (presentations, images)
```

#### Cross-Filesystem Search
```
command> Search for files about Python
[ask] → Found 5 matching files:
  - /local/python_tutorial.pdf (87% match)
  - /interact/python_project.docx (82% match)
  - /local/django_notes.txt (76% match)
  ...
```

#### Cross-Filesystem Organization
```
command> Move all PDFs from local to interact
[organize] → Successfully moved 3 files from 'local' to 'interact':
  - report.pdf
  - guide.pdf
  - manual.pdf
```

---

## Filesystem Adapters

### Local Adapter
- **Use case**: Local filesystem operations
- **Setup**: Specify `--root` directory or use default temp directory
- **Features**: Full read/write, rename, move, delete

### CodeSandbox Adapter
- **Use case**: CodeSandbox environment files
- **Setup**: Requires CodeSandbox SDK authentication
- **Features**: Read/write via SDK, download/upload support

### Interact Adapter
- **Use case**: Remote browser files (downloads, uploads)
- **Setup**: Requires active Interact session
- **Features**: Read/write via API, stream support

### Google Drive Adapter
- **Use case**: Google Drive files and folders
- **Setup**: Requires OAuth 2.0 authentication (see `GOOGLE_DRIVE_SETUP.md`)
- **Features**: Read/write, Google Workspace format conversion

---

## Advanced Features

### Voice Mode

Enable hands-free operation with voice input/output:

```bash
python -m sandboxes.file_manager.file_manager_sandbox --voice
```

Once running, press Enter on an empty line or type `r` to record voice input.

### Clarification Requests

Both sandboxes support mid-execution clarification requests. The manager can ask for additional information when needed:

```
command> Organize my files
[organize] →
🔔 Clarification needed: How would you like the files organized? By type, date, or custom criteria?
clarification> By file type, please
[organize] → Organized 15 files into 4 folders by type: documents/, images/, spreadsheets/, presentations/
```

### Steering Controls

During execution, you can:
- **Pause**: Type `/pause` to pause execution
- **Interject**: Type `/interject <message>` to add context
- **Cancel**: Type `/cancel` to stop execution
- **Resume**: Type `/resume` to continue after pause

### Scenario Generation

Use the `us` (update scenario) command to generate synthetic file environments:

```
command> us Create a realistic corporate file structure with 20 documents across departments
```

The LLM will:
1. Generate appropriate file names and types
2. Create realistic file contents
3. Organize files hierarchically
4. Populate metadata (dates, sizes, etc.)

---

## Testing Workflows

### Basic FileManager Testing

1. **Setup**: Start sandbox with local adapter
   ```bash
   python -m sandboxes.file_manager.file_manager_sandbox --adapter local --root /tmp/test_files
   ```

2. **Import Files**: Use `us` command to generate sample files
   ```
   command> us Create 10 sample business documents
   ```

3. **Query**: Ask questions about files
   ```
   command> What's in the quarterly report?
   command> Which files mention revenue?
   ```

4. **Organize**: Reorganize files
   ```
   command> Rename Q1_report.pdf to 2024_Q1_report.pdf
   command> Move all spreadsheets to /data folder
   ```

### Cross-Filesystem Testing

1. **Setup**: Start with multiple filesystems
   ```bash
   python -m sandboxes.file_manager.global_file_manager_sandbox --filesystems local,interact
   ```

2. **Populate**: Generate files across filesystems
   ```
   command> us Create sample files in both local and interact filesystems
   ```

3. **Cross-Query**: Search across filesystems
   ```
   command> Find all files containing "budget"
   command> Compare files in local vs interact
   ```

4. **Cross-Organize**: Move/sync files
   ```
   command> Copy all reports from local to interact
   command> Sync presentation files between filesystems
   ```

---

## Troubleshooting

### Common Issues

**Issue**: "No adapter configured"
- **Solution**: Ensure you're using a valid adapter name: `local`, `codesandbox`, `interact`, `google_drive`

**Issue**: "File not found"
- **Solution**: Check file exists with exact name/path. Use namespace prefix for GlobalFileManager (e.g., `/local/file.txt`)

**Issue**: "Permission denied"
- **Solution**: Some files are protected. Check `is_protected()` status. Use appropriate credentials for remote adapters.

**Issue**: Voice mode not working
- **Solution**: Ensure microphone permissions and Deepgram API key is configured in `.env`

**Issue**: Google Drive authentication fails
- **Solution**: Follow setup instructions in `GOOGLE_DRIVE_SETUP.md` for OAuth 2.0 credentials

---

## Loom Walkthrough

**Coming Soon**: A comprehensive video walkthrough demonstrating:
- FileManager sandbox basics
- GlobalFileManager cross-filesystem operations
- Voice mode capabilities
- Advanced steering and clarification features
- Real-world testing scenarios

---

## Development Notes

### Architecture

Both sandboxes follow the standard Unity manager sandbox pattern:

1. **Intent Classification**: LLM judge routes user input to appropriate method
2. **Tool Loop**: AsyncToolLoop orchestrates LLM + tool execution
3. **Clarifications**: Optional bidirectional clarification queues
4. **Steering**: SteerableToolHandle for mid-flight control

### Adding New Adapters

To add support for a new filesystem adapter:

1. Create adapter in `unity/file_manager/fs_adapters/`
2. Create manager in `unity/file_manager/managers/`
3. Add to `--adapter` choices in FileManager sandbox
4. Add to `--filesystems` options in GlobalFileManager sandbox

### Testing Philosophy

These sandboxes support:
- **Exploratory testing**: Ad-hoc queries and operations
- **Scenario testing**: LLM-generated synthetic environments
- **Integration testing**: Cross-component interactions
- **User acceptance testing**: Real-world workflows

---

## Related Documentation

- [FileManager API](../../unity/file_manager/README.md)
- [GlobalFileManager API](../../unity/file_manager/README.md)
- [Google Drive Setup](../../unity/file_manager/GOOGLE_DRIVE_SETUP.md)
- [Filesystem Adapters](../../unity/file_manager/fs_adapters/README.md)
- [Parser Documentation](../../unity/file_manager/parser/README.md)

---

## Support

For issues or questions:
1. Check existing documentation
2. Review test files in `tests/test_file_manager/`
3. Examine similar manager sandboxes for patterns
4. Consult Unity framework documentation

---

**Last Updated**: 2025-10-12
**Maintainer**: Unity Team
