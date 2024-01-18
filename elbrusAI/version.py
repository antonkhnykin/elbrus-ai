from elbrusAI.api_export import elbrusAI_export

# Unique source of truth for the version number.
__version__ = "0.0.1"


@elbrusAI_export("elbrusAI.version")
def version():
    return __version__