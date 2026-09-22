"""Host the audit viewer (one self-contained HTML, built by ``scripts/build_viewer.py``) on Modal.

Redeploy after every rebuild — the file is baked into the image; stop the app first so a warm
old container stops serving: ``modal app stop -y joracle-viewer``.

    JORACLE_SITE_HTML=outputs/live/site/index.html modal deploy scripts/serve_viewer_modal.py

Needs ``modal`` (``pip install -e ".[serve]"``). Any static host works instead: the page is
one file with no external requests beyond Google Fonts.
"""

import os
from pathlib import Path

import modal

DEFAULT = "outputs/live/site/index.html"
PAGE = Path(os.environ.get("JORACLE_SITE_HTML", DEFAULT)).resolve()
if modal.is_local() and not PAGE.exists():
    raise SystemExit(f"page not found: {PAGE} (build it first with scripts/build_viewer.py)")

app = modal.App(os.environ.get("JORACLE_VIEWER_APP", "joracle-viewer"))
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.12")
    .add_local_file(str(PAGE), "/site/index.html")
)


@app.function(image=image, cpu=0.25, memory=512, scaledown_window=300)
@modal.asgi_app()
def web():  # type: ignore[no-untyped-def]
    from fastapi import FastAPI
    from fastapi.responses import FileResponse

    api = FastAPI()

    @api.get("/")
    def index():  # type: ignore[no-untyped-def]
        return FileResponse("/site/index.html", media_type="text/html")

    return api
