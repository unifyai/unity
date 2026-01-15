from .file import (
    # New type names
    DocumentFields,
    DocumentRow,
    Document,
    FileRecord,
    FileRecordRow,
    FileTableRefRow,
    # Backward compatibility aliases
    FileContentFields,
    FileContentRow,
    FileContent,
)
from .filesystem import (
    FileSystemCapabilities,
    FileReference,
    FolderReference,
)
from unity.file_manager.file_parsers.types.formats import FileFormat, MimeType
from .config import (
    FilePipelineConfig,
    ParseConfig,
    IngestConfig,
    EmbeddingsConfig,
    FileEmbeddingSpec,
    TableEmbeddingSpec,
    FileBusinessContextSpec,
    TableBusinessContextSpec,
    BusinessContextsConfig,
    OutputConfig,
    DiagnosticsConfig,
)
from .ingest import (
    BaseIngestedFile,
    IngestedPDF,
    IngestedDocx,
    IngestedDoc,
    IngestedXlsx,
    IngestedCsv,
    IngestedMinimal,
    IngestedFileUnion,
    FileResultType,
    ContentRef,
    TableRef,
    FileMetrics,
    PipelineStatistics,
    IngestPipelineResult,
)
from .describe import (
    ColumnInfo,
    ContextSchema,
    DocumentInfo,
    TableInfo,
    FileStorageMap,
)

__all__ = [
    # New type names (preferred)
    "DocumentFields",
    "DocumentRow",
    "Document",
    # Legacy types
    "FileRecord",
    "FileRecordRow",
    "FileTableRefRow",
    # Backward compatibility aliases (deprecated)
    "FileContentFields",
    "FileContentRow",
    "FileContent",
    "FileSystemCapabilities",
    "FileReference",
    "FolderReference",
    "FileFormat",
    "MimeType",
    "FilePipelineConfig",
    "ParseConfig",
    "IngestConfig",
    "EmbeddingsConfig",
    "FileEmbeddingSpec",
    "TableEmbeddingSpec",
    "FileBusinessContextSpec",
    "TableBusinessContextSpec",
    "BusinessContextsConfig",
    "OutputConfig",
    "DiagnosticsConfig",
    # Ingested models
    "BaseIngestedFile",
    "IngestedPDF",
    "IngestedDocx",
    "IngestedDoc",
    "IngestedXlsx",
    "IngestedCsv",
    "IngestedMinimal",
    "IngestedFileUnion",
    "FileResultType",
    "ContentRef",
    "TableRef",
    "FileMetrics",
    "PipelineStatistics",
    "IngestPipelineResult",
    # Describe API types
    "ColumnInfo",
    "ContextSchema",
    "DocumentInfo",
    "TableInfo",
    "FileStorageMap",
]
