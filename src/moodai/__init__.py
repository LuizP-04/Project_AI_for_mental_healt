"""moodai: local, privacy-first decision support for mood disorder hypotheses.

The tool suggests hypotheses, with evidence, for a mental health professional
to assess. It offers decision support, not a diagnosis.

Importing this package turns on offline mode (see ``moodai.offline``), so no
library can download models or send telemetry during an analysis.
"""

from importlib.metadata import version

from moodai.offline import enforce_offline

__version__ = version("moodai")

enforce_offline()
