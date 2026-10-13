"""Load markdown docs and split them into section-level chunks."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    heading: str
    text: str  # body only; used for answers and citations

    @property
    def search_text(self) -> str:
        return f"{self.title}. {self.heading}. {self.text}"


def load_chunks(docs_dir) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        m = re.search(r"^# (.+)$", raw, flags=re.M)
        title = m.group(1).strip() if m else path.stem
        for i, part in enumerate(re.split(r"^## ", raw, flags=re.M)[1:]):
            heading, _, body = part.partition("\n")
            body = body.strip()
            if body:
                chunks.append(Chunk(f"{path.stem}#{i}", path.stem, title, heading.strip(), body))
    return chunks
