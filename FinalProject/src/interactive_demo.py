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
    CSS = """
    Screen {
        padding: 1 2;
        background: #1b1b1b;
        color: #e0e0e0;
    }

    /* Panels */
#sidebar, #right_pane {
        background: #222;
        border: solid #373737;
        padding: 1;
    }

#sidebar {
        width: 32%;
        min-width: 30;
    }

#right_pane {
        margin-left: 1;
    }

    /* Section headers */
#header_alg, #header_ds, #header_inp, #header_res {
        text-style: bold;
        padding: 1 0;
        background: #2c2c2c;
        border-bottom: solid #444;
        content-align: center middle;
    }

    /* Lists */
#alg_set, #datasets_list {
        margin-top: 1;
        background: #1f1f1f;
        border: solid #333;
        padding: 1;
    }

    /* Pattern input */
#pattern_input {
        margin: 1 0;
        background: #111;
        border: solid #444;
    }

    /* Button */
#search_btn {
        margin-bottom: 1;
        background: #303030;
        border: solid #555;
    }

    /* Status bar */
#status {
        padding: 1;
        background: #242424;
        border: solid #444;
        margin-bottom: 1;
    }

    /* Results area */
#results_view {
        margin-top: 1;
        background: #151515;
        border: solid #333;
        padding: 1;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("Search Algorithm", id="header_alg")
                rs_items = [RadioButton(name, name=name) for name in ALGORITHMS.keys()]
                rs_items[0].value = True
                yield RadioSet(*rs_items, id="alg_set")

                yield Static("Datasets", id="header_ds")
                items = (
                    [ListItem(Label(p.name)) for p in DATASET_FILES]
                    if DATASET_FILES
                    else [ListItem(Label("(no .txt files found)"))]
                )
                yield ListView(*items, id="datasets_list")

            with Vertical(id="right_pane"):
                yield Static("Pattern Input", id="header_inp")
                yield Input(
                    placeholder="Type pattern and press Enter or click Search",
                    id="pattern_input",
                )
                yield Button("Search", id="search_btn")
                yield Static("Status: idle", id="status")

                yield Static("Results", id="header_res")
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
        for idx in matches[:20]:
            self.results_view.mount(Static(self._make_preview(text, idx, len(pattern))))

        elapsed = (t1 - t0) * 1000
        status.update(
            "\n".join(
                [
                    "[b]Status:[/b]     done",
                    f"[b]Dataset:[/b]    {ds_path.name}",
                    f"[b]Matches:[/b]    [green]{total}[/green]",
                    f"[b]Time:[/b]       [yellow]{elapsed:.2f} ms[/yellow]",
                ]
            )
        )


if __name__ == "__main__":
    SearchApp().run()
