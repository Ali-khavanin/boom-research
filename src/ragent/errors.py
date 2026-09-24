class RagentError(Exception):
    """Base error for expected user-facing failures."""


class MetricError(RagentError):
    """A graph transition repeatedly failed its termination metric."""


class ProviderError(RagentError):
    """An LLM provider could not satisfy a request."""


class GraphError(RagentError):
    """A graph or graph-bound tool is invalid."""


class BudgetError(RagentError):
    """A token or cost budget was exhausted."""


class PreflightError(RagentError):
    """Free admission checks failed before a research run."""


class ExportError(RagentError):
    """A configured local or remote export could not be verified."""


class VerificationError(RagentError):
    """Persisted run evidence did not satisfy its recorded contract."""
