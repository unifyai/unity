"""What an ingestion asks for: where rows come from, and where they land.

Two small discriminated unions rather than one wide config. Source and target vary
independently, and every combination of them means something real:

===============  ==================  ====================================
source           target              meaning
===============  ==================  ====================================
``RowsSource``   ``TableTarget``     an API, integration or manual upload
``TableSource``  ``TableTarget``     reshape or copy stored rows
``FilesSource``  ``CollectionTarget``  documents kept whole, with their tables
``FilesSource``  ``TableTarget``     spreadsheets or CSVs into one data table
``FolderSource`` either              the same, at batch scale
===============  ==================  ====================================

Keeping them separate is what lets the actor pick each on its own terms: it knows
what it is holding (the source) and what it wants to do with it (the target),
and never has to reason about a single config whose fields half apply.

**Where the work runs is not expressible here, on purpose.** There is no mode,
tier or worker field, because the choice is not a judgement about intent -- it
follows from measured size and from what the deployment has running, both of
which the manager knows and a caller does not. Offering the knob would invite a
guess in place of a measurement, and the guess would be wrong in the direction
that hurts: parsing a large file in the assistant's own process.

Nothing is lost by withholding it. Both tiers write the same artifacts and the
same checkpoints, so a run is resumable and recoverable either way, and asking
about one reads identically.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from unify.data_manager.types.ingest import PostIngestConfig

# =============================================================================
# Sources -- where the rows come from
# =============================================================================


class RowsSource(BaseModel):
    """Rows already in hand.

    The source for anything the plan has just fetched or built: a third-party API
    response, a connected-app pull via ``primitives.integrations.*``, a computed
    summary, or values a user supplied directly.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["rows"] = "rows"
    rows: List[Dict[str, Any]] = Field(
        description="Row dicts to store. Keys become column names.",
    )

    @field_validator("rows")
    @classmethod
    def _reject_empty(cls, value: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # An empty ingestion is almost always a bug upstream -- an API returned
        # nothing and the plan did not notice. Failing here names the real problem
        # instead of reporting a successful run that stored nothing.
        if not value:
            raise ValueError(
                "RowsSource.rows is empty. Check the upstream call returned data "
                "before submitting, or skip the ingestion.",
            )
        return value


class FilesSource(BaseModel):
    """Specific files, by path.

    Covers uploads, chat and email attachments, exports written to disk, and any
    file already visible to the assistant. Parsing is handled by the file
    pipeline, which selects a backend per format -- documents (PDF, DOCX),
    tabular formats (CSV, XLSX) and the rest are each handled properly, so the
    caller does not choose a parser.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["files"] = "files"
    paths: List[str] = Field(description="File paths to parse and store.")

    @field_validator("paths")
    @classmethod
    def _reject_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("FilesSource.paths is empty; nothing to ingest.")
        return value


class FolderSource(BaseModel):
    """Everything matching a pattern under a directory.

    The batch shape: a drive or SharePoint pull, an export directory, a month of
    reports. Prefer this over listing hundreds of paths in ``FilesSource`` when the
    set is open-ended -- naming a directory and a pattern says the membership is
    whatever matches at the time, which is both shorter and more accurate than a
    snapshot of paths taken earlier.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["folder"] = "folder"
    path: str = Field(description="Directory to walk.")
    pattern: str = Field(
        default="*",
        description="Glob applied to file names, e.g. '*.xlsx'.",
    )
    recursive: bool = True


class TableSource(BaseModel):
    """Rows already stored in a context.

    For reshaping, narrowing or copying what is already there -- deriving a
    smaller table a canvas can bind to cheaply, or re-keying an accumulated log
    into a current-state view.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["table"] = "table"
    context: str = Field(description="Context path to read from.")
    filter: Optional[str] = Field(
        default=None,
        description="Server-side filter expression; omit to read everything.",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to carry over; omit for all.",
    )


IngestionSource = Annotated[
    Union[RowsSource, FilesSource, FolderSource, TableSource],
    Field(discriminator="kind"),
]


# =============================================================================
# Targets -- where it lands
# =============================================================================

# A collection name is interpolated into a context path, so it has to be a single
# safe path segment. Rejecting separators and traversal here is what stops one
# collection from being written into another's namespace.
_COLLECTION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")


class TableTarget(BaseModel):
    """One queryable table at a context path you choose.

    Use this when the point is to *query columns*: filter, reduce, join, or bind a
    canvas to it. The context path is explicit and stable, which is what makes it
    bindable.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["table"] = "table"
    context: str = Field(
        description=(
            "Context path to write, e.g. 'Data/HubSpotDeals'. Stable and "
            "explicit, so a canvas binding or a later query can name it."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="What the table holds. Worth setting: it is what a later reader sees.",
    )
    unique_keys: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Column -> type map identifying a row. Supplying it makes a re-run an "
            "upsert rather than an append, which is what a current-state view "
            "wants. Omit to accumulate a time series."
        ),
    )
    fields: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Explicit column schema; omit to infer from the rows.",
    )
    infer_untyped_fields: bool = Field(
        default=False,
        description=(
            "Let the backend infer types for columns with no declared type. "
            "Useful for spreadsheet data where a column arrives as strings."
        ),
    )

    @field_validator("context")
    @classmethod
    def _validate_context(cls, value: str) -> str:
        cleaned = value.strip().strip("/")
        if not cleaned:
            raise ValueError("TableTarget.context cannot be empty.")
        if ".." in cleaned:
            raise ValueError(
                f"TableTarget.context {value!r} contains '..'; give a plain context path.",
            )
        return cleaned


class CollectionTarget(BaseModel):
    """A named group of files, each keeping its own identity.

    Two jobs, and the second is easy to miss. It keeps documents whole -- read
    them, search them, cite them. It is also **the way to ingest many files as
    one run**: with ``extract_tables`` on, every table inside every file becomes
    its own queryable context, so fifteen spreadsheets become fifteen queryable
    tables under one handle.

    Prefer it over one ``TableTarget`` per file whenever the files are a set.
    ``TableTarget`` names a single context, so N tables means N runs, and that
    trades away the single run id that makes the batch observable -- per-file
    state, rows and destinations all live on one status
    (``status.files``) when the batch is one run. The file pipeline
    writes the parsed content and each extracted table beneath the collection,
    and the run reports the exact context paths it produced, so a canvas can bind
    to a table out of a spreadsheet without anyone hardcoding the layout.

    Leaving ``name`` unset keeps each file in its own auto-assigned namespace,
    which is right for unrelated one-off files. Setting it groups a related set --
    a quarter of reports, one client's documents -- under a stable name that can
    be bound to and added to later.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["collection"] = "collection"
    name: Optional[str] = Field(
        default=None,
        description=(
            "Collection name, a single path segment. Omit to give each file its "
            "own auto-assigned namespace."
        ),
    )
    extract_tables: bool = Field(
        default=True,
        description="Store tables found inside documents as their own queryable contexts.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not _COLLECTION_NAME.match(cleaned):
            raise ValueError(
                f"CollectionTarget.name {value!r} is not a single safe path segment. "
                "Use letters, digits, spaces, '_' or '-' -- no '/' and no '..'.",
            )
        # An all-numeric name would alias the namespace an unnamed file is assigned,
        # which is its numeric record id. The storage layer distinguishes a shared
        # collection from a single file's namespace by exactly that comparison, so a
        # collection called "42" holding the file whose id is 42 would be read as
        # unshared -- and re-storing that one file would then clear every *other*
        # file's rows in the collection rather than just its own. Reserving the
        # numeric namespace makes the collision impossible instead of unlikely.
        if cleaned.isdigit():
            raise ValueError(
                f"CollectionTarget.name {value!r} is all digits, which collides with "
                "the namespace an unnamed file is given. Include at least one "
                "letter, e.g. 'batch-42'.",
            )
        return cleaned


IngestionTarget = Annotated[
    Union[TableTarget, CollectionTarget],
    Field(discriminator="kind"),
]


# =============================================================================
# Common options -- identical whatever the source or target is
# =============================================================================


class EmbedSpec(BaseModel):
    """Which text to make semantically searchable.

    Applies the same way to rows and to file content, so switching between them
    does not change how embedding is requested.
    """

    model_config = ConfigDict(extra="forbid")

    columns: List[str] = Field(description="Text columns to embed.")
    strategy: Literal["along", "after"] = Field(
        default="along",
        description=(
            "'along' embeds as rows are inserted, so partial results are usable "
            "sooner. 'after' embeds once insertion finishes."
        ),
    )


class IngestionRequest(BaseModel):
    """One complete ingestion, before it is run.

    The unit that is validated, recorded and then executed. Holding it as a value
    is what lets the same request be retried or resumed later without the caller
    reconstructing it.
    """

    model_config = ConfigDict(extra="forbid")

    source: IngestionSource
    target: IngestionTarget
    embed: Optional[EmbedSpec] = None
    post_ingest: Optional[PostIngestConfig] = Field(
        default=None,
        description="Derived columns computed once the rows are in.",
    )
    destination: Optional[str] = Field(
        default=None,
        description=(
            "Ownership root: 'personal' (the default) or 'team:<id>'. The privacy "
            "floor is personal; ask rather than guess toward the wider audience."
        ),
    )

    @field_validator("target")
    @classmethod
    def _reject_meaningless_pairs(cls, target: Any, info: Any) -> Any:
        """Refuse combinations that cannot mean anything, naming the fix.

        Only one pairing is genuinely impossible: rows and stored tables have no
        documents to keep whole, so a collection has nothing to put in it. Every
        other combination is meaningful, including files into a single table.
        """
        source = (info.data or {}).get("source")
        if source is None:
            return target
        source_kind = getattr(source, "kind", None)
        target_kind = getattr(target, "kind", None)
        if target_kind == "collection" and source_kind in {"rows", "table"}:
            raise ValueError(
                f"A {source_kind} source has no documents to keep whole, so "
                "CollectionTarget does not apply. Use TableTarget(context=...) "
                "to store these rows as a queryable table.",
            )
        return target
