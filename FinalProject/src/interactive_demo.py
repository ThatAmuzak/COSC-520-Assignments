import time
from pathlib import Path
from typing import Callable, List

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

DATASET_DIR = Path.cwd() / "data/gutenberg_books"
DATASET_FILES = sorted(DATASET_DIR.glob("*.txt")) if DATASET_DIR.exists() else []

ALGORITHMS = {"KMP": kmp_search, "Rabin-Karp": rabin_karp, "Boyer-Moore": boyer_moore}


class SearchApp(App):
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical():
                items = (
                    [ListItem(Label(p.name)) for p in DATASET_FILES]
                    if DATASET_FILES
                    else [ListItem(Label("(no .txt files found)"))]
                )
                yield ListView(*items, id="datasets_list")
                rs_items = [RadioButton(name, name=name) for name in ALGORITHMS.keys()]
                rs_items[0].value = True
                yield RadioSet(*rs_items, id="alg_set")
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

    def on_mount(self):
        self.query_one(Input).focus()

    def get_selected_dataset(self):
        sel = self.query_one(ListView).index
        if sel is None or sel < 0 or sel >= len(DATASET_FILES):
            return DATASET_FILES[0] if DATASET_FILES else None
        return DATASET_FILES[sel]

    def get_selected_algorithm(self) -> Callable[[str, str], List[int]]:
        rs = self.query_one(RadioSet)
        for rb in rs.query(RadioButton):
            if rb.value:
                return ALGORITHMS[rb.name]
        return next(iter(ALGORITHMS.values()))

    def _load_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def _make_preview(self, text: str, start: int, pat_len: int, ctx: int = 20) -> str:
        s = max(0, start - ctx)
        e = min(len(text), start + pat_len + ctx)
        snippet = text[s:e].replace("\n", " ")
        if s > 0:
            snippet = "..." + snippet
        if e < len(text):
            snippet += "..."
        rs, re = start - s + 3, start - s + pat_len + 3
        if 0 <= rs < len(snippet):
            snippet = snippet[:rs] + "[[" + snippet[rs:re] + "]]" + snippet[re:]
        return snippet

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "search_btn":
            self.do_search()

    def on_input_submitted(self, event: Input.Submitted):
        self.do_search()

    def do_search(self):
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
        text = self._load_text(ds_path)

        t0 = time.perf_counter()
        matches = algo(pattern, text)
        t1 = time.perf_counter()

        self.results_view.remove_children()
        total = len(matches)
        self.results_view.mount(Static("================================="))
        self.results_view.mount(
            Static(f"Results: {total} matches, time: {(t1 - t0) * 1000:.2f} ms")
        )
        self.results_view.mount(Static("================================="))
        for idx in matches[:30]:
            self.results_view.mount(Static(self._make_preview(text, idx, len(pattern))))
        if total > 10:
            self.results_view.mount(Static("================================="))
            self.results_view.mount(Static(f"... and {total - 10} more matches"))

        status.update(
            f"Status: done. dataset={ds_path.name} matches={total} time={
                (t1 - t0) * 1000:.2f}ms"
        )


if __name__ == "__main__":
    SearchApp().run()
