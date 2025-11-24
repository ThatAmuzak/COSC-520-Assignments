"""
TUI/interactive_demo.py
Minimal interactive demo of KMP, Rabin-Karp, Boyer-Moore using textual.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from algorithms import boyer_moore, kmp_search, rabin_karp
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.scroll_view import ScrollView
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RadioButton,
    RadioSet,
    Static,
)

PROJECT_ROOT = Path.cwd()
POSSIBLE_DATASET_PATHS = [PROJECT_ROOT / "src" / "datasets"]


def find_datasets_dir() -> Optional[Path]:
    for p in POSSIBLE_DATASET_PATHS:
        if p.exists() and p.is_dir():
            return p
    return None


def load_dataset_files(datasets_dir: Path) -> List[Path]:
    return sorted(datasets_dir.glob("*.txt"))


def import_algorithms_module() -> Optional[object]:
    try:
        import importlib

        return importlib.import_module("src.algorithms")
    except Exception:
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "src.algorithms", str(Path.cwd() / "src" / "algorithms.py")
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                return mod
        except Exception:
            return None


def get_algorithms() -> Dict[str, Callable[[str, str], List[int]]]:
    return {"KMP": kmp_search, "Rabin-Karp": rabin_karp, "Boyer-Moore": boyer_moore}


class SearchApp(App):
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.datasets_dir = find_datasets_dir()
        self.dataset_files = (
            load_dataset_files(self.datasets_dir) if self.datasets_dir else []
        )
        self.alg_module = import_algorithms_module()
        self.algorithms = get_algorithms()
        self.current_text: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical():
                items = (
                    [ListItem(Label(p.name)) for p in self.dataset_files]
                    if self.dataset_files
                    else [ListItem(Label("(no .txt files found)"))]
                )
                yield ListView(*items, id="datasets_list")
                if not self.algorithms:
                    yield Label("(no algorithms detected)")
                else:
                    rs_items = [
                        RadioButton(name, name=name) for name in self.algorithms.keys()
                    ]
                    rs = RadioSet(*rs_items, id="alg_set")
                    rs_items[0].value = True
                    yield rs
            with Vertical():
                yield Label("Pattern to search:")
                yield Input(
                    placeholder="Type pattern and press Enter or click Search",
                    id="pattern_input",
                )
                yield Button("Search", id="search_btn")
                yield Static("Status: idle", id="status")
                self.results_view = ScrollView(id="results_view")
                yield self.results_view
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def get_selected_dataset(self) -> Optional[Path]:
        lv = self.query_one(ListView)
        sel = lv.index
        if sel is None or sel < 0 or sel >= len(self.dataset_files):
            return self.dataset_files[0] if self.dataset_files else None
        return self.dataset_files[sel]

    def get_selected_algorithm(self) -> Optional[Callable[[str, str], List[int]]]:
        if not self.algorithms:
            return None
        rs = self.query_one(RadioSet)
        for rb in rs.query(RadioButton):
            if rb.value:
                return self.algorithms.get(rb.name)
        return next(iter(self.algorithms.values()))

    def _load_text(self, path: Path) -> str:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        self.current_text = txt
        return txt

    def _make_preview(
        self, text: str, start: int, pat_len: int, context_chars: int = 40
    ) -> str:
        s = max(0, start - context_chars)
        e = min(len(text), start + pat_len + context_chars)
        snippet = text[s:e].replace("\n", " ")
        if s > 0:
            snippet = "..." + snippet
        if e < len(text):
            snippet = snippet + "..."
        rel_start = start - s
        rel_end = rel_start + pat_len
        if 0 <= rel_start < len(snippet):
            snippet = (
                snippet[:rel_start]
                + "[["
                + snippet[rel_start:rel_end]
                + "]]"
                + snippet[rel_end:]
            )
        return snippet

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "search_btn":
            self.do_search()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.do_search()

    def do_search(self) -> None:
        status = self.query_one("#status", Static)
        pattern = self.query_one(Input).value or ""
        if not pattern:
            status.update("Status: enter a pattern")
            return
        ds_path = self.get_selected_dataset()
        if not ds_path:
            status.update("Status: no dataset")
            return
        algo = self.get_selected_algorithm()
        if not algo:
            status.update("Status: no algorithm")
            return
        text = self._load_text(ds_path)
        t0 = time.perf_counter()
        matches = algo(pattern, text)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000

        # Clear previous results manually
        self.results_view.remove_children()
        total = len(matches) if matches else 0
        self.results_view.mount(
            Static(f"Results: {total} matches, time: {elapsed_ms:.2f} ms")
        )
        for idx in matches[:10]:
            self.results_view.mount(Static(self._make_preview(text, idx, len(pattern))))
        if total > 10:
            self.results_view.mount(Static(f"... and {total - 10} more matches"))
        status.update(
            f"Status: done. dataset={ds_path.name} matches={total} time={
                elapsed_ms:.2f}ms"
        )


if __name__ == "__main__":
    SearchApp().run()
