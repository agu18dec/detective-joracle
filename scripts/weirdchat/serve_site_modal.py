"""Host the WeirdChat explain viewer (``index.html`` + ``data/*.json``) on Modal.

The site is a directory, not one file: ``build_site.py`` writes a small index and one data file
per pattern that the page fetches on demand. Redeploy after every rebuild — the directory is
baked into the image; stop the app first so a warm old container stops serving:
``modal app stop -y weirdchat-site``.

    WEIRDCHAT_SITE=outputs/weirdchat/site uvx --python 3.12 modal@latest deploy \\
        scripts/weirdchat/serve_site_modal.py

Any static host works instead (``python -m http.server`` in the site dir).
"""

import os
from pathlib import Path

import modal

DEFAULT = "outputs/weirdchat/site"
SITE = Path(os.environ.get("WEIRDCHAT_SITE", DEFAULT)).resolve()
if modal.is_local() and not (SITE / "index.html").exists():
    raise SystemExit(
        f"site not found: {SITE} (build it first with scripts/weirdchat/build_site.py)"
    )

app = modal.App(os.environ.get("WEIRDCHAT_SITE_APP", "weirdchat-site"))
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.115.12")
    .add_local_dir(str(SITE), "/site")
)


@app.function(image=image, cpu=0.25, memory=512, scaledown_window=300)
@modal.asgi_app()
def web():  # type: ignore[no-untyped-def]
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    api = FastAPI()

    @api.get("/")
    def index():  # type: ignore[no-untyped-def]
        return FileResponse("/site/index.html", media_type="text/html")

    api.mount("/data", StaticFiles(directory="/site/data"), name="data")
    return api
