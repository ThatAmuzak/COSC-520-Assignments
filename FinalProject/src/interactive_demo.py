import time
from pathlib import Path
from typing import Callable, List

from algorithms import boyer_moore, kmp_search, rabin_karp
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
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

ALGORITHMS = {
    "KMP": kmp_search,
    "Rabin-Karp": rabin_karp,
    "Boyer-Moore": boyer_moore,
}


class SearchApp(App):
    CSS = """
    Screen {
        padding: 1 2;
        background: #121212;
        color: #e0e0e0;
    }

    #sidebar, #right_pane {
        background: #1c1c1c;
        border: solid #444;
        padding: 1;
    }

    #sidebar {
        width: 32%;
        min-width: 30;
    }

    #right_pane {
        margin-left: 1;
    }

    #top_section {
        height: 30%;
    }

    #bottom_section {
        height: 60%;
    }

    #search_and_status {
        layout: horizontal;
        margin: 0;
    }

    #pattern_input, #status {
        width: 1fr;
    }

    /* Section headers with deep colors */
    #header_alg, #header_ds {
        text-style: bold;
        padding: 1 0;
        background: #8e2de2;
        color: #fff;
        border-bottom: solid #555;
        content-align: center middle;
    }

    #header_inp {
        text-style: bold;
        padding: 1 0;
        background: #ff6a00;
        color: #fff;
        content-align: center middle;
    }

    #header_res {
        text-style: bold;
        padding: 1 0;
        background: #00c6ff;
        color: #111;
        content-align: center middle;
    }

    #alg_set, #datasets_list {
        margin-top: 1;
        background: #222;
        border: solid #333;
        padding: 1;
    }

    #pattern_input {
        margin: 0;
        background: #1a1a1a;
        border: solid #555;
        color: #fff;
    }

    #search_btn {
        margin-bottom: 1;
        background: #ff416c;
        border: solid #777;
        color: #fff;
        width: 100%;
    }

    #status {
        padding: 1;
        background: #333333;
        border: solid #555;
        color: #fff;
        margin-bottom: 1;
    }

    #results_view {
        margin-top: 1;
        background: #1b1b1b;
        border: solid #333;
        padding: 1;
        color: #ddd;
    }

    ListView > Static {
        border-bottom: dashed #444;
        padding: 0 1;
    }

    RadioButton {
        background: #2a2a2a;
        color: #ddd;
    }

    RadioButton.--selected {
        background: #6a0dad;
        color: #fff;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("a", "next_algorithm", "Next Algorithm"),
        ("d", "next_dataset", "Next Dataset"),
        ("A", "previous_algorithm", "Previous Algorithm"),
        ("D", "previous_dataset", "Previous Dataset"),
        ("i", "focus_input", "Focus Input"),
    ]

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
                with Vertical(id="top_section"):
                    with Horizontal():
                        yield Static("Pattern Input", id="header_inp")
                    with Horizontal(id="search_and_status"):
                        yield Input(
                            placeholder="Type pattern and press Enter or click Search",
                            id="pattern_input",
                        )
                        yield Static("Status: idle", id="status")

                yield Button("Search", id="search_btn")
                with Vertical(id="bottom_section"):
                    yield Static("Results", id="header_res")
                    self.results_view = ListView(id="results_view")
                    yield self.results_view

        yield Footer()

    def on_mount(self):
        self.query_one(Input).focus()
        self.results_view.remove_children()

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

    def _make_preview(self, text: str, start: int, pat_len: int, ctx: int = 40) -> str:
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
            self.results_view.remove_children()
            status.update("[i]Status[/i]: enter a pattern")
            return

        ds_path = self.get_selected_dataset()
        if not ds_path:
            status.update("[i]Status[/i]: no dataset")
            return

        algo = self.get_selected_algorithm()
        text = self._load_text(ds_path)

        t0 = time.perf_counter()
        matches = algo(pattern, text)
        t1 = time.perf_counter()

        self.results_view.remove_children()
        total = len(matches)
        if total == 0:
            self.results_view.mount(Static("No matches found"))
        else:
            for idx in matches[:50]:
                self.results_view.mount(
                    Static(self._make_preview(text, idx, len(pattern)))
                )

        elapsed = (t1 - t0) * 1000
        status.update(
            "\n".join(
                [
                    "[i]Status:[/i]     [b]Done[/b]",
                    f"[i]Dataset:[/i]    [b]{ds_path.name}[/b]",
                    f"[i]Matches:[/i]    [b][lime]{total}[/lime][/b]",
                    f"[i]Time:[/i]       [b][gold]{elapsed:.2f} ms[/gold][/b]",
                ]
            )
        )

    def action_focus_input(self):
        self.query_one(Input).focus()

    def action_next_algorithm(self):
        rs = self.query_one(RadioSet)
        buttons = list(rs.query(RadioButton))
        idx = next((i for i, b in enumerate(buttons) if b.value), 0)
        buttons[idx].value = False
        buttons[(idx + 1) % len(buttons)].value = True

    def action_next_dataset(self):
        lv = self.query_one("#datasets_list", ListView)
        if not DATASET_FILES:
            return
        next_idx = 0 if lv.index is None else (lv.index + 1) % len(DATASET_FILES)
        lv.index = next_idx

    def action_previous_algorithm(self):
        rs = self.query_one(RadioSet)
        buttons = list(rs.query(RadioButton))
        idx = next((i for i, b in enumerate(buttons) if b.value), 0)
        buttons[idx].value = False
        buttons[(idx - 1) % len(buttons)].value = True

    def action_previous_dataset(self):
        lv = self.query_one("#datasets_list", ListView)
        if not DATASET_FILES:
            return
        prev_idx = (
            (lv.index - 1) % len(DATASET_FILES)
            if lv.index is not None
            else len(DATASET_FILES) - 1
        )
        lv.index = prev_idx


if __name__ == "__main__":
    SearchApp().run()
