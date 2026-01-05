# Data Files

This directory contains or links to the data files used by the Repairs Analysis Agent.

## Data Files

The following Excel files should be available (either directly or via symlink):

1. **MDH Repairs Data July - Nov 25 - DL V1.xlsx**
   - Contains: Repairs job data
   - Tables: `Raised_01-07-2025_to_30-11-2025`
   - Columns: JobTicketReference, OperativeWhoCompletedJob, FirstTimeFix, NoAccess, FollowOn, etc.

2. **MDH Telematics Data July - Nov 25 - DL V1.xlsx**
   - Contains: Vehicle telematics data
   - Tables: `July_2025`, `August_2025`, `September_2025`, `October_2025`, `November_2025`
   - Columns: Vehicle, Business distance, Trip travel time, EndLocation, etc.

## Setup

The data files are typically stored in `intranet/repairs/` and ingested into FileManager
using the `repairs_file_pipeline_config_5m.json` configuration.

### Creating Symlinks (optional)

```bash
# From intranet/repairs_agent/data/
ln -s ../../repairs/MDH\ Repairs\ Data\ July\ -\ Nov\ 25\ -\ DL\ V1.xlsx .
ln -s ../../repairs/MDH\ Telematics\ Data\ July\ -\ Nov\ 25\ -\ DL\ V1.xlsx .
```

## File Locations

The actual file paths used in the metrics are configured in `metrics/constants.py`:

```python
REPAIRS_FILE = "/home/hmahmood24/unity/intranet/repairs/MDH Repairs Data July - Nov 25 - DL V1.xlsx"
TELEMATICS_FILE = "/home/hmahmood24/unity/intranet/repairs/MDH Telematics Data July - Nov 25 - DL V1.xlsx"
```

## FileManager Context Paths

After ingestion, the data is accessible via FileManager context paths like:
- `Files/Repairs/MDH_Repairs_Data.../Tables/Raised_01-07-2025_to_30-11-2025`
- `Files/Telematics/MDH_Telematics_Data.../Tables/July_2025`

Use `tables_overview()` to discover the actual context paths at runtime.
