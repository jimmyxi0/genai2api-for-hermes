import re


def slugify(name: str) -> str:
    """Simple slugify to create URL/ID friendly path segments."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip('-')
    if not s:
        s = "unknown"
    return s


def resolve_context7_library_id(library_name: str, version: str | None = None, organization: str = "org") -> dict:
    """Resolve a Context7 library name to a stable library_id.

    Returns a dict with keys:
      - library_id: string path like /org/<slug>[/<version>]
      - library_name: original input name
      - version: optional version if provided
    """
    slug = slugify(library_name)
    library_id = f"/{organization}/{slug}"
    if version:
        library_id = f"{library_id}/{version}"
    return {
        "library_id": library_id,
        "library_name": library_name,
        "version": version,
    }
