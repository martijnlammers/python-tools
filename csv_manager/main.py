"""
CSV Manager — Textual TUI with two tabs:
  Files  – browse, create & delete CSV files
  Data   – view, search, add/delete rows & columns (auto-saved)
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)
from textual.screen import ModalScreen

# ── Default paths (overridable via CSVManagerApp class attributes) ────
DEFAULT_DATA_DIR = Path("data/csv")
DEFAULT_CONFIG_PATH = Path("data/config.json")


# ── Pure helpers (no side effects, easy to unit-test) ─────────────────

def _safe_id(name: str) -> str:
    """Turn an arbitrary string into a valid Textual CSS identifier."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name).strip("-") or "col"


DEFAULT_CONFIG = {
    "theme": "textual-dark",
    "last_file": "",
    "last_tab": "files",
}


def _load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Read JSON config from *path*, back-filling any missing keys with defaults."""
    try:
        cfg = json.loads(path.read_text())
    except Exception:
        cfg = {}
    updated = False
    for key, value in DEFAULT_CONFIG.items():
        if key not in cfg:
            cfg[key] = value
            updated = True
    if updated:
        _save_config(cfg, path)
    return cfg


def _save_config(cfg: dict, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Write *cfg* as JSON to *path*, creating parent dirs as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2))
    except OSError:
        pass  # non-fatal: config is a convenience, not critical


def filter_dataframe(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return rows of *df* matching *query* (pure function, no UI).

    Supports:
      - Multiple terms (space-separated, AND logic)
      - column:value  to search a specific column
      - !term          to exclude rows matching a term
      - "exact phrase" for literal multi-word match
    """
    if not query:
        return df

    try:
        tokens = shlex.split(query)
    except ValueError:
        tokens = query.split()

    mask = pd.Series(True, index=df.index)
    for token in tokens:
        negate = False
        if token.startswith("!") and len(token) > 1:
            negate = True
            token = token[1:]
        if ":" in token and not token.startswith(":"):
            col_part, val_part = token.split(":", 1)
            matches = [c for c in df.columns if c.lower() == col_part.lower()]
            if matches:
                term_mask = df[matches[0]].astype(str).str.contains(
                    val_part, case=False, na=False, regex=False,
                )
            else:
                term_mask = pd.Series(False, index=df.index)
        else:
            term_mask = df.apply(
                lambda row, t=token: row.astype(str)
                .str.contains(t, case=False, na=False, regex=False)
                .any(),
                axis=1,
            )
        mask &= ~term_mask if negate else term_mask
    return df[mask]


async def _run_git(*args: str) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode or 0, out.decode().strip(), err.decode().strip()


class EditCellScreen(ModalScreen[Optional[str]]):
    """Modal popup for editing a single cell value."""

    CSS = """
    EditCellScreen {
        align: center middle;
    }
    #edit-dialog {
        width: 60;
        height: auto;
        max-height: 12;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #edit-dialog Label {
        margin-bottom: 1;
    }
    #edit-cell-input {
        width: 100%;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self, col_name: str, row_num: int, current_val: str) -> None:
        super().__init__()
        self._col_name = col_name
        self._row_num = row_num
        self._current_val = current_val

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-dialog"):
            yield Label(f"Edit Row {self._row_num} — {self._col_name}")
            yield Input(value=self._current_val, id="edit-cell-input")

    def on_mount(self) -> None:
        self.query_one("#edit-cell-input", Input).focus()

    @on(Input.Submitted, "#edit-cell-input")
    def _on_submit(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AddColumnScreen(ModalScreen[Optional[Tuple[str, str]]]):
    """Modal popup to add a new column with a name and optional default value."""

    CSS = """
    AddColumnScreen {
        align: center middle;
    }
    #add-col-dialog {
        width: 60;
        height: auto;
        max-height: 40%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #add-col-dialog Label {
        margin-bottom: 1;
    }
    .col-field-label {
        margin: 0;
        padding: 0 1;
        color: $text-muted;
    }
    .col-field-input {
        width: 100%;
        margin-bottom: 1;
    }
    #add-col-submit {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    def compose(self) -> ComposeResult:
        with Vertical(id="add-col-dialog"):
            yield Label("Add New Column")
            yield Static("Column name", classes="col-field-label")
            yield Input(placeholder="Enter column name…", id="col-name-input", classes="col-field-input")
            yield Static("Default value (optional)", classes="col-field-label")
            yield Input(placeholder="Leave empty for blank", id="col-default-input", classes="col-field-input")
            yield Button("Add Column", variant="primary", id="add-col-submit")

    def on_mount(self) -> None:
        self.query_one("#col-name-input", Input).focus()

    @on(Input.Submitted, "#col-name-input")
    def _on_name_submit(self, event: Input.Submitted) -> None:
        self.query_one("#col-default-input", Input).focus()

    @on(Input.Submitted, "#col-default-input")
    def _on_default_submit(self, event: Input.Submitted) -> None:
        self._submit()

    @on(Button.Pressed, "#add-col-submit")
    def _on_submit_btn(self) -> None:
        self._submit()

    def _submit(self) -> None:
        name = self.query_one("#col-name-input", Input).value.strip()
        default = self.query_one("#col-default-input", Input).value
        if not name:
            self.app._warn("Column name cannot be empty!")
            return
        self.dismiss((name, default))

    def action_cancel(self) -> None:
        self.dismiss(None)


class RemoveColumnScreen(ModalScreen[Optional[str]]):
    """Modal popup to pick a column to remove."""

    CSS = """
    RemoveColumnScreen {
        align: center middle;
    }
    #rm-col-dialog {
        width: 60;
        height: auto;
        max-height: 60%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #rm-col-dialog Label {
        margin-bottom: 1;
    }
    .rm-col-btn {
        width: 100%;
        min-width: 1;
        height: 1;
        margin: 0;
        padding: 0 1;
        border: none;
        background: transparent;
        text-style: none;
    }
    .rm-col-btn:hover {
        background: $accent 20%;
    }
    .rm-col-btn:focus {
        background: $accent 30%;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("j,down", "cursor_down", "Down", show=False),
        Binding("k,up", "cursor_up", "Up", show=False),
    ]

    def __init__(self, columns: list[str]) -> None:
        super().__init__()
        self._columns = columns

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="rm-col-dialog"):
            yield Label("Remove Column")
            for col in self._columns:
                yield Button(col, id=f"rm-col-{_safe_id(col)}", classes="rm-col-btn")

    def on_mount(self) -> None:
        btns = list(self.query(".rm-col-btn").results(Button))
        if btns:
            btns[0].focus()

    def _get_focused_index(self) -> int:
        btns = list(self.query(".rm-col-btn").results(Button))
        for i, btn in enumerate(btns):
            if btn.has_focus:
                return i
        return 0

    def action_cursor_down(self) -> None:
        btns = list(self.query(".rm-col-btn").results(Button))
        if not btns:
            return
        idx = (self._get_focused_index() + 1) % len(btns)
        btns[idx].focus()

    def action_cursor_up(self) -> None:
        btns = list(self.query(".rm-col-btn").results(Button))
        if not btns:
            return
        idx = (self._get_focused_index() - 1) % len(btns)
        btns[idx].focus()

    @on(Button.Pressed, ".rm-col-btn")
    def _on_col_pressed(self, event: Button.Pressed) -> None:
        col_name = event.button.label.plain
        self.dismiss(col_name)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AddFileScreen(ModalScreen[Optional[Tuple[str, str]]]):
    """Modal popup to create a new CSV file."""

    CSS = """
    AddFileScreen {
        align: center middle;
    }
    #add-file-dialog {
        width: 60;
        height: auto;
        max-height: 40%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #add-file-dialog Label {
        margin-bottom: 1;
    }
    .file-field-label {
        margin: 0;
        padding: 0 1;
        color: $text-muted;
    }
    .file-field-input {
        width: 100%;
        margin-bottom: 1;
    }
    #add-file-submit {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="add-file-dialog"):
            yield Label("Create New CSV")
            yield Static("File name (without .csv)", classes="file-field-label")
            yield Input(placeholder="Enter file name\u2026", id="file-name-input", classes="file-field-input")
            yield Static("Columns (comma-separated, optional)", classes="file-field-label")
            yield Input(placeholder="e.g. name, age, score", id="file-cols-input", classes="file-field-input")
            yield Button("Create File", variant="primary", id="add-file-submit")

    def on_mount(self) -> None:
        self.query_one("#file-name-input", Input).focus()

    @on(Input.Submitted, "#file-name-input")
    def _on_name_submit(self, event: Input.Submitted) -> None:
        cols_input = self.query_one("#file-cols-input", Input)
        cols_input.focus()
        cols_input.scroll_visible()

    @on(Input.Submitted, "#file-cols-input")
    def _on_cols_submit(self, event: Input.Submitted) -> None:
        self._submit()

    @on(Button.Pressed, "#add-file-submit")
    def _on_submit_btn(self) -> None:
        self._submit()

    def _submit(self) -> None:
        name = self.query_one("#file-name-input", Input).value.strip()
        cols = self.query_one("#file-cols-input", Input).value.strip()
        if not name:
            self.app._warn("File name cannot be empty!")
            return
        self.dismiss((name, cols))

    def action_cancel(self) -> None:
        self.dismiss(None)


class AddRowScreen(ModalScreen[Optional[dict]]):
    """Modal popup with an input for every column to add a new row."""

    CSS = """
    AddRowScreen {
        align: center middle;
    }
    #add-row-dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #add-row-dialog Label {
        margin-bottom: 1;
    }
    .field-label {
        margin: 0;
        padding: 0 1;
        color: $text-muted;
    }
    .field-input {
        width: 100%;
        margin-bottom: 1;
    }
    #add-row-submit {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self, columns: list[str]) -> None:
        super().__init__()
        self._columns = columns

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="add-row-dialog"):
            yield Label("Add New Row")
            for col in self._columns:
                yield Static(col, classes="field-label")
                default = datetime.now().strftime("%Y-%m-%d %H:%M") if col.lower() == "timestamp" else ""
                yield Input(
                    value=default,
                    placeholder=f"Enter {col}…",
                    id=f"field-{_safe_id(col)}",
                    classes="field-input",
                )
            yield Button("Add Row", variant="primary", id="add-row-submit")

    def on_mount(self) -> None:
        # Focus the first input
        first = self.query(".field-input").first(Input)
        if first:
            first.focus()

    @on(Input.Submitted, ".field-input")
    def _on_field_submit(self, event: Input.Submitted) -> None:
        """Tab-like: move focus to next input, or submit if last."""
        inputs = list(self.query(".field-input").results(Input))
        try:
            idx = inputs.index(event.input)
            if idx < len(inputs) - 1:
                inputs[idx + 1].focus()
            else:
                self._submit_row()
        except ValueError:
            pass

    @on(Button.Pressed, "#add-row-submit")
    def _on_submit_btn(self) -> None:
        self._submit_row()

    def _submit_row(self) -> None:
        row = {}
        for col in self._columns:
            inp = self.query_one(f"#field-{_safe_id(col)}", Input)
            row[col] = inp.value
        self.dismiss(row)

    def action_cancel(self) -> None:
        self.dismiss(None)


class FilesContent(Vertical):
    """Container for Files tab."""


class DataContent(Vertical):
    """Container for Data tab."""


class CSVManagerApp(App):
    """Main application.  Override *data_dir* / *config_path* for testing."""

    TITLE = "CSV Manager"

    data_dir: Path = DEFAULT_DATA_DIR
    config_path: Path = DEFAULT_CONFIG_PATH

    def _info(self, text: str) -> None:
        """Show an informational toast."""
        self.notify(text, timeout=2)

    def _warn(self, text: str) -> None:
        """Show a warning toast."""
        self.notify(text, severity="warning", timeout=2)

    def _error(self, text: str) -> None:
        """Show an error toast."""
        self.notify(text, severity="error", timeout=2)

    CSS = """
    /* ── Global interactive styles ────────────────────────── */
    Input {
        border: none;
        height: 3;
        padding: 1 1;
        margin: 0;
    }
    Input:hover {
        background: $accent 20%;
    }
    Input:focus {
        background: $accent 30%;
    }
    Button:hover {
        background: $accent 30%;
    }
    DataTable > .datatable--cursor {
        background: $accent 30%;
    }

    /* ── Layout ───────────────────────────────────────────── */
    #file-table {
        height: auto;
        max-height: 15;
    }
    #data-status {
        background: $primary-background;
        padding: 0 1;
        text-style: bold;
        margin: 0;
    }
    #row-status {
        color: $text-muted;
        padding: 0 1;
    }
    #data-content {
        height: 1fr;
    }
    #toolbar {
        height: auto;
        margin: 0;
        padding: 0;
    }
    #search-input {
        width: 1fr;
    }

    /* ── Custom footer ────────────────────────────────────── */
    #custom-footer {
        dock: bottom;
        height: 1;
        background: $panel;
    }
    #footer-left {
        width: 1fr;
        height: 1;
        padding: 0 1;
    }
    #footer-right {
        width: auto;
        height: 1;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("delete", "delete_rows", "delete", show=False),
        Binding("a", "add", "add", show=False),
        Binding("o", "open_file", "open", show=False),
        Binding("s", "focus_search", "search", show=False),
        Binding("c", "focus_add_col", "add col", show=False),
        Binding("r", "focus_remove_col", "rm col", show=False),
        Binding("q", "quit", "quit", show=False),
        Binding("g", "git_sync", "git commit", show=False),
        Binding("t", "toggle_theme", "theme", show=False),
        Binding("1", "switch_tab('files')", "files", show=False),
        Binding("2", "switch_tab('data')", "data", show=False),
    ]

    current_file: reactive[Path | None] = reactive(None)
    current_df: pd.DataFrame | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="tabs"):

            with TabPane("[b $accent]1[/] files", id="files"):
                with FilesContent():
                    yield Label("Your Files \u2014 click a file to open it in the Data tab")
                    yield DataTable(id="file-table", zebra_stripes=True, cursor_type="cell")

            with TabPane("[b $accent]2[/] data", id="data"):
                with DataContent():
                    with VerticalScroll(id="data-content"):
                        # \u2500\u2500 Status bar \u2500\u2500
                        yield Static("No file loaded", id="data-status")
                        # \u2500\u2500 Toolbar: search + column management \u2500\u2500
                        with Horizontal(id="toolbar"):
                            yield Input(placeholder="Search rows\u2026", id="search-input")
                        # \u2500\u2500 Data table \u2500\u2500
                        yield DataTable(zebra_stripes=True, id="data-table", cursor_type="cell")
                        yield Static("", id="row-status")

        with Horizontal(id="custom-footer"):
            yield Static("", id="footer-left")
            yield Static("", id="footer-right")

    def on_key(self, event) -> None:
        """Global key handler: escape blurs inputs, arrows navigate tables."""
        if event.key == "escape":
            focused = self.focused
            if isinstance(focused, Input):
                focused.blur()
                event.prevent_default()
                event.stop()
                return
        # Block left/right from switching tabs (allow in inputs/tables)
        if event.key in ("left", "right"):
            focused = self.focused
            if not isinstance(focused, (Input, DataTable)):
                event.prevent_default()
                event.stop()
                return
        # On Files/Data tab, arrow up/down should navigate the table
        if event.key in ("up", "down"):
            try:
                active = self.query_one("#tabs", TabbedContent).active
            except Exception:
                return
            if active == "files":
                ft = self.query_one("#file-table", DataTable)
                ft.show_cursor = True
                if not ft.has_focus:
                    ft.focus()
                    event.prevent_default()
                    event.stop()
            elif active == "data":
                dt = self.query_one("#data-table", DataTable)
                if not dt.has_focus:
                    dt.focus()
                    event.prevent_default()
                    event.stop()

    def on_mount(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._error(f"Cannot create data dir: {exc}")
        cfg = _load_config(self.config_path)
        if "theme" in cfg:
            self.theme = cfg["theme"]
        ft = self.query_one("#file-table", DataTable)
        ft.add_columns("Name", "Columns", "Rows")
        ft.show_cursor = False
        self._refresh_file_list()
        self._update_footer()
        # Restore last open file
        if "last_file" in cfg:
            path = self.data_dir / cfg["last_file"]
            if path.exists():
                self._load_file(path)
        # Restore last active tab
        if "last_tab" in cfg:
            self.query_one("#tabs", TabbedContent).active = cfg["last_tab"]

    # ── helpers ──────────────────────────────────────────────────────────

    def _refresh_file_list(self) -> None:
        ft = self.query_one("#file-table", DataTable)
        with self.batch_update():
            ft.clear()
            csv_files = sorted(self.data_dir.glob("*.csv"))
            for f in csv_files:
                try:
                    df = pd.read_csv(f, dtype=str, keep_default_na=False)
                    cols = str(len(df.columns))
                    rows = str(len(df))
                except Exception:
                    cols = "0"
                    rows = "0"
                ft.add_row(f.name, cols, rows, key=f.stem)

    def _find_file(self, stem: str) -> Path | None:
        for f in self.data_dir.glob("*.csv"):
            if f.stem == stem:
                return f
        return None

    def _load_file(self, path: Path, *, silent: bool = False) -> None:
        self.current_file = path
        cfg = _load_config(self.config_path)
        cfg["last_file"] = path.name
        _save_config(cfg, self.config_path)
        try:
            self.current_df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            self.current_df = pd.DataFrame()

        with self.batch_update():
            rows = len(self.current_df) if self.current_df is not None else 0
            cols = len(self.current_df.columns) if self.current_df is not None else 0
            self.query_one("#data-status", Static).update(
                f" {path.name}  \u2502  {rows} rows  \u2502  {cols} columns"
            )
            self.query_one("#row-status", Static).update("")

            table: DataTable = self.query_one("#data-table", DataTable)
            table.clear(columns=True)
            if self.current_df is not None and (
                not self.current_df.empty or len(self.current_df.columns) > 0
            ):
                table.add_columns(*[str(c) for c in self.current_df.columns])
                for idx, r in self.current_df.iterrows():
                    table.add_row(*[str(v) for v in r.values])

            self.query_one("#search-input", Input).value = ""
        if not silent:
            self._info(f"Loaded {path.name} ({rows} rows, {cols} columns)")

    def _apply_search(self, query: str) -> None:
        """Filter the data table using :func:`filter_dataframe`."""
        table: DataTable = self.query_one("#data-table", DataTable)
        table.clear(columns=True)
        if self.current_df is None:
            return
        df = self.current_df
        filtered = filter_dataframe(df, query)

        if len(df.columns) > 0:
            table.add_columns(*[str(c) for c in df.columns])
            for _, r in filtered.iterrows():
                table.add_row(*[str(v) for v in r.values])
        self.query_one("#row-status", Static).update(
            f"Showing {len(filtered)} of {len(df)} rows" if query else ""
        )

    def _save_current(self) -> None:
        """Auto-save the current file silently."""
        if self.current_file and self.current_df is not None:
            self.current_df.to_csv(self.current_file, index=False)
            rows = len(self.current_df)
            cols = len(self.current_df.columns)
            self.query_one("#data-status", Static).update(
                f" {self.current_file.name}  \u2502  {rows} rows  \u2502  {cols} columns"
            )
            self._refresh_file_list()

    # ── tab persistence ──────────────────────────────────────────────────

    @on(TabbedContent.TabActivated, "#tabs")
    def _on_tab_changed(self, event: TabbedContent.TabActivated) -> None:
        cfg = _load_config(self.config_path)
        cfg["last_tab"] = event.pane.id
        _save_config(cfg, self.config_path)
        self._update_footer()

    def _update_footer(self) -> None:
        """Render the custom footer with context bindings left, common right."""
        try:
            active = self.query_one("#tabs", TabbedContent).active
        except Exception:
            active = "files"

        left: list[tuple[str, str]] = []
        if active in ("files", "data"):
            left.append(("del", "delete"))
            left.append(("a", "add"))
        if active == "files":
            left.append(("o", "open"))
        if active == "data":
            left.append(("s", "search"))
            left.append(("c", "add col"))
            left.append(("r", "rm col"))

        right: list[tuple[str, str]] = [
            ("q", "quit"),
            ("g", "git commit"),
            ("t", "theme"),
        ]

        def _fmt(pairs: list[tuple[str, str]]) -> str:
            return "  ".join(f"[b $accent]{k}[/] {v}" for k, v in pairs)

        self.query_one("#footer-left", Static).update(_fmt(left))
        self.query_one("#footer-right", Static).update(_fmt(right))

    # ── actions ──────────────────────────────────────────────────────────

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab_id

    def action_open_file(self) -> None:
        """Open the file under the cursor in the file table."""
        ft = self.query_one("#file-table", DataTable)
        try:
            row_key, _ = ft.coordinate_to_cell_key(ft.cursor_coordinate)
            stem = str(row_key.value)
            path = self._find_file(stem)
            if path:
                self._load_file(path)
                self.query_one("#tabs", TabbedContent).active = "data"
        except Exception:
            pass

    def action_focus_search(self) -> None:
        """Focus the search input on the Data tab."""
        self.query_one("#tabs", TabbedContent).active = "data"
        self.query_one("#search-input", Input).focus()

    def action_focus_add_col(self) -> None:
        """Open the Add Column modal."""
        if self.current_df is None or self.current_file is None:
            self._warn("Open a CSV file first!")
            return

        def _on_add_col_result(result: tuple[str, str] | None) -> None:
            if result is not None and self.current_df is not None and self.current_file is not None:
                col_name, default_val = result
                if col_name in self.current_df.columns:
                    self._warn(f"Column '{col_name}' already exists!")
                    return
                self.current_df[col_name] = default_val
                self._save_current()
                self._load_file(self.current_file, silent=True)
                self._info(f"Added column '{col_name}'")

        self.push_screen(AddColumnScreen(), callback=_on_add_col_result)

    def action_focus_remove_col(self) -> None:
        """Open the Remove Column modal."""
        if self.current_df is None or self.current_file is None:
            self._warn("Open a CSV file first!")
            return
        columns = list(self.current_df.columns)
        if not columns:
            self._warn("No columns to remove!")
            return

        def _on_rm_col_result(col_name: str | None) -> None:
            if col_name is not None and self.current_df is not None and self.current_file is not None:
                if col_name in self.current_df.columns:
                    self.current_df = self.current_df.drop(columns=[col_name])
                    self._save_current()
                    self._load_file(self.current_file, silent=True)
                    self._info(f"Removed column '{col_name}'")

        self.push_screen(RemoveColumnScreen(columns), callback=_on_rm_col_result)

    # ── button handlers ──────────────────────────────────────────────────

    @on(DataTable.CellSelected, "#file-table")
    def _on_file_cell_selected(self, event: DataTable.CellSelected) -> None:
        ft = self.query_one("#file-table", DataTable)
        row_key, _ = ft.coordinate_to_cell_key(event.coordinate)
        stem = str(row_key.value)
        path = self._find_file(stem)
        if path:
            self._load_file(path)
            self.query_one("#tabs", TabbedContent).active = "data"

    @on(Input.Changed, "#search-input")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._apply_search(event.value.strip())

    def action_add(self) -> None:
        """Context-aware add: add file on Files tab, add row on Data tab."""
        try:
            active = self.query_one("#tabs", TabbedContent).active
        except Exception:
            return
        if active == "files":
            self._add_file()
            return
        self._add_row()

    def _add_file(self) -> None:
        """Open the Add File modal to create a new CSV."""
        def _on_add_file_result(result: tuple[str, str] | None) -> None:
            if result is None:
                return
            name, raw_cols = result
            path = self.data_dir / f"{name}.csv"
            if path.exists():
                self._warn(f"{path.name} already exists!")
                return
            columns = [c.strip() for c in raw_cols.split(",") if c.strip()] if raw_cols else []
            columns = ["timestamp"] + [c for c in columns if c != "timestamp"]
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            self._refresh_file_list()
            self._info(f"Created {path.name}")
            self._load_file(path)
            self.query_one("#tabs", TabbedContent).active = "data"

        self.push_screen(AddFileScreen(), callback=_on_add_file_result)

    def _add_row(self) -> None:
        """Open the Add Row modal to insert a new row."""
        if self.current_df is None or self.current_file is None:
            self._warn("Open a CSV file first!")
            return
        columns = list(self.current_df.columns)
        if not columns:
            self._warn("No columns defined yet!")
            return

        def _on_add_result(row: dict | None) -> None:
            if row is not None and self.current_df is not None and self.current_file is not None:
                new_row = pd.DataFrame([row])
                self.current_df = pd.concat([self.current_df, new_row], ignore_index=True)
                self._save_current()
                self._load_file(self.current_file, silent=True)
                self._info(f"Added row {len(self.current_df)}")

        self.push_screen(AddRowScreen(columns), callback=_on_add_result)

    def action_delete_rows(self) -> None:
        """Delete selected rows or file depending on active tab / focused table."""
        ft = self.query_one("#file-table", DataTable)
        dt = self.query_one("#data-table", DataTable)

        # Determine which table to act on: prefer focused, fall back to active tab
        try:
            active = self.query_one("#tabs", TabbedContent).active
        except Exception:
            return
        if ft.has_focus or (not dt.has_focus and active == "files"):
            # Delete file
            try:
                row_key, _ = ft.coordinate_to_cell_key(ft.cursor_coordinate)
                stem = str(row_key.value)
                path = self._find_file(stem)
                if path:
                    path.unlink()
                    self._refresh_file_list()
                    if self.current_file and self.current_file.stem == stem:
                        self.current_file = None
                        self.current_df = None
                        self.query_one("#data-status", Static).update("No file loaded")
                        self.query_one("#data-table", DataTable).clear(columns=True)
                    self._info(f"Deleted {stem}.csv")
            except Exception:
                pass
            return

        if not dt.has_focus and active != "data":
            return
        if self.current_df is None or self.current_file is None:
            return
        table = self.query_one("#data-table", DataTable)
        row_idx = table.cursor_coordinate.row
        if 0 <= row_idx < len(self.current_df):
            self.current_df = self.current_df.drop(self.current_df.index[row_idx]).reset_index(drop=True)
            self._save_current()
            self._load_file(self.current_file, silent=True)
            self._info(f"Deleted row {row_idx + 1}")

    @on(DataTable.CellSelected, "#data-table")
    def _on_data_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Open the edit-cell modal for the clicked cell."""
        if self.current_df is None or self.current_file is None:
            return
        row_idx = event.coordinate.row
        col_idx = event.coordinate.column

        if 0 <= row_idx < len(self.current_df) and 0 <= col_idx < len(self.current_df.columns):
            col_name = self.current_df.columns[col_idx]
            current_val = str(self.current_df.iloc[row_idx, col_idx])
            if current_val == "nan":
                current_val = ""
            self.query_one("#row-status", Static).update(
                f"Row {row_idx + 1} / {len(self.current_df)}  \u2502  Column: {col_name}"
            )

            def _on_edit_result(new_val: str | None) -> None:
                if new_val is not None and self.current_df is not None and self.current_file is not None:
                    self.current_df.iloc[row_idx, col_idx] = new_val
                    self._save_current()
                    table = self.query_one("#data-table", DataTable)
                    table.update_cell_at((row_idx, col_idx), new_val)
                    self._info(f"Updated Row {row_idx + 1}, {col_name}")

            self.push_screen(
                EditCellScreen(col_name, row_idx + 1, current_val),
                callback=_on_edit_result,
            )

    # ── input shortcuts ──────────────────────────────────────────────────

    def action_toggle_theme(self) -> None:
        themes = list(self.available_themes.keys())
        current = themes.index(self.theme) if self.theme in themes else 0
        self.theme = themes[(current + 1) % len(themes)]
        cfg = _load_config(self.config_path)
        cfg["theme"] = self.theme
        _save_config(cfg, self.config_path)
        self._info(f"Theme: {self.theme}")

    # ── git commit & push ────────────────────────────────────────────────

    def action_git_sync(self) -> None:
        self._info("Committing & pushing...")
        self.run_worker(self._git_sync_worker(), exclusive=True)

    async def _git_sync_worker(self) -> None:
        data_path = str(self.data_dir.parent)  # data/ folder
        try:
            await _run_git("add", data_path)

            rc, _, _ = await _run_git("diff", "--cached", "--quiet")
            if rc == 0:
                self._info("Already up to date")
                self._refresh_file_list()
                return

            commit_message = f"csv-manager: sync data ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
            rc, _, err = await _run_git("commit", "-m", commit_message)
            if rc != 0:
                self._error(f"Commit failed: {err}")
                return

            rc, _, err = await _run_git("push")
            if rc != 0:
                self._error(f"Push failed: {err}")
                return

            self._info("Committed & pushed to git")
            self._refresh_file_list()
        except FileNotFoundError:
            self._error("Git not found - is git installed?")
        except Exception as exc:
            self._error(str(exc))


if __name__ == "__main__":
    CSVManagerApp().run()
