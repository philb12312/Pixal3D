import os
from pathlib import Path


def _ensure_default_cc() -> None:
	if os.name == "nt" or os.environ.get("CC"):
		return

	zig_cc = Path(__file__).resolve().parents[1] / "scripts" / "zig-cc.sh"
	if zig_cc.exists():
		os.environ["CC"] = str(zig_cc)


_ensure_default_cc()

from . import models
from . import modules
from . import pipelines
from . import renderers
from . import representations
from . import utils
