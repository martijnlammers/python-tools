"""
CSV Manager — Textual TUI with three tabs:
  Files  – browse, create & delete CSV files
  Data   – view, add/delete rows, add columns, save
  Plot   – pick axes & chart type, open a matplotlib figure
"""

from __future__ import annotations

import os
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rich.text import Text

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

DATA_DIR = Path("data/csv")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path("data/config.json")


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


class CSVManagerApp(App):

    TITLE = "CSV Manager"

    CSS = """
    #file-table {
        height: auto;
        max-height: 15;
    }
    #file-table > .datatable--cursor {
        background: $accent 30%;
    }
    #data-status {
        background: $primary-background;
        padding: 0 1;
        text-style: bold;
    }
    #data-toolbar {
        height: 3;
    }
    #row-status {
        color: $text-muted;
        padding: 0 1;
    }
    #search-input {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("g", "git_sync", "Git Sync", show=True),
        Binding("t", "toggle_theme", "Theme", show=True),
    ]

    current_file: reactive[Path | None] = reactive(None)
    current_df: pd.DataFrame | None = None
    _rebuild_counter: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent("Files", "Data", "Plot", id="tabs"):

            with TabPane("Files", id="files"):
                with Vertical():
                    yield Label("New CSV \u2014 enter a name and columns, then press Enter")
                    with Horizontal():
                        yield Input(placeholder="File name (without .csv)", id="name-input")
                        yield Input(placeholder="Columns (comma-separated)", id="columns-input")
                    yield Label("")
                    yield Label("Your Files — click a file to open it in the Data tab")
                    yield DataTable(id="file-table", zebra_stripes=True, cursor_type="cell")

            with TabPane("Data", id="data"):
                with VerticalScroll():
                    # ── Status bar ──
                    yield Static("No file loaded", id="data-status")
                    # ── Toolbar ──
                    with Horizontal(id="data-toolbar"):
                        yield Button("Save", variant="success", id="save-btn", disabled=True)
                        yield Button("Delete Row", variant="error", id="del-row-btn", disabled=True)
                        yield Button("Delete Column", variant="error", id="del-col-btn", disabled=True)
                        yield Input(placeholder="Search rows...", id="search-input")
                    # ── Data table ──
                    yield DataTable(zebra_stripes=True, id="data-table", cursor_type="cell")
                    yield Static("", id="row-status")
                    # ── Add row ──
                    yield Label("New Row")
                    with Horizontal(id="row-inputs-container"):
                        pass
                    yield Button("Add Row", variant="primary", id="add-row-btn", disabled=True)
                    # ── Column management ──
                    yield Label("Column Management")
                    with Horizontal():
                        yield Input(placeholder="New column name", id="new-col-input", disabled=True)
                        yield Button("Add Column", variant="warning", id="add-col-btn", disabled=True)
                    with Horizontal():
                        yield Select([], prompt="Select column", id="col-select", allow_blank=True, disabled=True)
                        yield Input(placeholder="New name", id="rename-col-input", disabled=True)
                        yield Button("Rename", variant="primary", id="rename-col-btn", disabled=True)

            with TabPane("Plot", id="plot"):
                with Vertical():
                    yield Label("Chart Configuration")
                    with Horizontal():
                        yield Select([], prompt="X axis", id="x-select", allow_blank=True)
                        yield Select([], prompt="Y axis", id="y-select", allow_blank=True)
                        yield Select(
                            [("Line", "line"), ("Bar", "bar"), ("Scatter", "scatter"), ("Area", "area")],
                            prompt="Chart type", id="chart-select", value="line",
                        )
                    yield Button("Generate Plot", variant="primary", id="plot-btn", disabled=True)
                    yield Label("")
                    yield Static("Open a CSV from the Files tab, then come here to plot.", id="plot-status")

        yield Footer()

    def on_key(self, event) -> None:
        if event.key == "escape":
            focused = self.focused
            if isinstance(focused, Input):
                focused.blur()
                event.prevent_default()
                event.stop()

    def on_mount(self) -> None:
        cfg = _load_config()
        if "theme" in cfg:
            self.theme = cfg["theme"]
        ft = self.query_one("#file-table", DataTable)
        ft.add_columns("Name", "Columns", "Rows", "Remove")
        self._refresh_file_list()

    # ── helpers ──────────────────────────────────────────────────────────

    def _refresh_file_list(self) -> None:
        ft = self.query_one("#file-table", DataTable)
        with self.batch_update():
            ft.clear()
            csv_files = sorted(DATA_DIR.glob("*.csv"))
            for f in csv_files:
                try:
                    df = pd.read_csv(f)
                    cols = str(len(df.columns))
                    rows = str(len(df))
                except Exception:
                    cols = "0"
                    rows = "0"
                ft.add_row(f.name, cols, rows, Text("Delete", style="bold red"), key=f.stem)

    def _find_file(self, stem: str) -> Path | None:
        for f in DATA_DIR.glob("*.csv"):
            if f.stem == stem:
                return f
        return None

    def _load_file(self, path: Path) -> None:
        self.current_file = path
        try:
            self.current_df = pd.read_csv(path)
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
                for _, r in self.current_df.iterrows():
                    table.add_row(*[str(v) for v in r.values])

            self.query_one("#save-btn", Button).disabled = False
            self.query_one("#del-row-btn", Button).disabled = False
            self.query_one("#del-col-btn", Button).disabled = False
            self.query_one("#add-row-btn", Button).disabled = False
            self.query_one("#new-col-input", Input).disabled = False
            self.query_one("#add-col-btn", Button).disabled = False
            self.query_one("#rename-col-input", Input).disabled = False
            self.query_one("#rename-col-btn", Button).disabled = False
            self.query_one("#col-select", Select).disabled = False
            self.query_one("#plot-btn", Button).disabled = False
            self.query_one("#search-input", Input).value = ""

            self._rebuild_row_inputs()
            self._refresh_plot_selects()
            self._refresh_col_select()
        self.notify(f"Loaded {path.name} ({rows} rows, {cols} columns)")

    def _rebuild_row_inputs(self) -> None:
        container = self.query_one("#row-inputs-container", Horizontal)
        with self.batch_update():
            container.remove_children()
            self._rebuild_counter += 1
            if self.current_df is not None:
                for idx, col in enumerate(self.current_df.columns):
                    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(col))
                    container.mount(Input(placeholder=str(col), id=f"ri__{self._rebuild_counter}_{idx}_{safe}"))

    def _refresh_plot_selects(self) -> None:
        if self.current_df is None or self.current_df.columns.empty:
            return
        cols = [(str(c), str(c)) for c in self.current_df.columns]
        self.query_one("#x-select", Select).set_options(cols)
        self.query_one("#y-select", Select).set_options(cols)

    def _refresh_col_select(self) -> None:
        if self.current_df is None or self.current_df.columns.empty:
            self.query_one("#col-select", Select).set_options([])
            return
        cols = [(str(c), str(c)) for c in self.current_df.columns]
        self.query_one("#col-select", Select).set_options(cols)

    def _apply_search(self, query: str) -> None:
        """Filter the data table to show only rows matching the search query."""
        table: DataTable = self.query_one("#data-table", DataTable)
        table.clear(columns=True)
        if self.current_df is None:
            return
        df = self.current_df
        if query:
            mask = df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)
            filtered = df[mask]
        else:
            filtered = df
        if len(df.columns) > 0:
            table.add_columns(*[str(c) for c in df.columns])
            for _, r in filtered.iterrows():
                table.add_row(*[str(v) for v in r.values])
        self.query_one("#row-status", Static).update(
            f"Showing {len(filtered)} of {len(df)} rows" if query else ""
        )

    def _save_current(self) -> None:
        if self.current_file and self.current_df is not None:
            self.current_df.to_csv(self.current_file, index=False)
            rows = len(self.current_df)
            cols = len(self.current_df.columns)
            self.query_one("#data-status", Static).update(
                f" {self.current_file.name}  \u2502  {rows} rows  \u2502  {cols} columns"
            )
            self.notify(f"Saved {self.current_file.name}")

    # ── actions ──────────────────────────────────────────────────────────

    def action_switch_tab(self, tab_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab_id

    # ── button handlers ──────────────────────────────────────────────────

    def _on_create(self) -> None:
        name_input: Input = self.query_one("#name-input", Input)
        col_input: Input = self.query_one("#columns-input", Input)
        name = name_input.value.strip()
        if not name:
            self.notify("Enter a file name first!", severity="warning")
            return
        path = DATA_DIR / f"{name}.csv"
        if path.exists():
            self.notify(f"{path.name} already exists!", severity="warning")
            return
        raw_cols = col_input.value.strip()
        columns = [c.strip() for c in raw_cols.split(",") if c.strip()] if raw_cols else []
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        name_input.value = ""
        col_input.value = ""
        self._refresh_file_list()
        self.notify(f"Created {path.name}")

    @on(DataTable.CellSelected, "#file-table")
    def _on_file_cell_selected(self, event: DataTable.CellSelected) -> None:
        ft = self.query_one("#file-table", DataTable)
        row_key, _ = ft.coordinate_to_cell_key(event.coordinate)
        stem = str(row_key.value)
        if event.coordinate.column == 3:
            # Remove column clicked — delete the file
            path = self._find_file(stem)
            if path:
                path.unlink()
                self._refresh_file_list()
                if self.current_file and self.current_file.stem == stem:
                    self.current_file = None
                    self.current_df = None
                    self.query_one("#data-status", Static).update("No file loaded")
                    self.query_one("#data-table", DataTable).clear(columns=True)
                self.notify(f"Deleted {stem}.csv")
        else:
            # Any other column — open the file
            path = self._find_file(stem)
            if path:
                self._load_file(path)
                self.query_one("#tabs", TabbedContent).active = "data"

    @on(Button.Pressed, "#save-btn")
    def _on_save(self) -> None:
        self._save_current()

    @on(Button.Pressed, "#del-row-btn")
    def _on_delete_row(self) -> None:
        if self.current_df is None or self.current_file is None:
            return
        table: DataTable = self.query_one("#data-table", DataTable)
        if table.cursor_row is not None and len(self.current_df) > 0:
            row_idx = table.cursor_row
            if 0 <= row_idx < len(self.current_df):
                self.current_df = self.current_df.drop(self.current_df.index[row_idx]).reset_index(drop=True)
                self._save_current()
                self._load_file(self.current_file)
                self.notify(f"Deleted row {row_idx + 1}")

    @on(Button.Pressed, "#add-row-btn")
    def _on_add_row(self) -> None:
        if self.current_df is None or self.current_file is None:
            return
        values = {}
        for idx, col in enumerate(self.current_df.columns):
            safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(col))
            try:
                inp: Input = self.query_one(f"#ri__{self._rebuild_counter}_{idx}_{safe}", Input)
                values[col] = inp.value
            except Exception:
                values[col] = ""
        self.current_df = pd.concat([self.current_df, pd.DataFrame([values])], ignore_index=True)
        self._save_current()
        self._load_file(self.current_file)
        self.notify("Row added!")

    @on(Button.Pressed, "#add-col-btn")
    def _on_add_column(self) -> None:
        if self.current_df is None or self.current_file is None:
            return
        col_input: Input = self.query_one("#new-col-input", Input)
        col_name = col_input.value.strip()
        if not col_name:
            self.notify("Enter a column name!", severity="warning")
            return
        if col_name in self.current_df.columns:
            self.notify(f"Column '{col_name}' already exists!", severity="warning")
            return
        self.current_df[col_name] = ""
        col_input.value = ""
        self._save_current()
        self._load_file(self.current_file)
        self.notify(f"Added column '{col_name}'")

    @on(Button.Pressed, "#del-col-btn")
    def _on_delete_column(self) -> None:
        if self.current_df is None or self.current_file is None:
            return
        table: DataTable = self.query_one("#data-table", DataTable)
        col_idx = table.cursor_column
        if col_idx is not None and 0 <= col_idx < len(self.current_df.columns):
            col_name = self.current_df.columns[col_idx]
            self.current_df = self.current_df.drop(columns=[col_name])
            self._save_current()
            self._load_file(self.current_file)
            self.notify(f"Deleted column '{col_name}'")
        else:
            self.notify("Select a cell in the column to delete", severity="warning")

    @on(Button.Pressed, "#rename-col-btn")
    def _on_rename_column(self) -> None:
        if self.current_df is None or self.current_file is None:
            return
        col_sel: Select = self.query_one("#col-select", Select)
        rename_input: Input = self.query_one("#rename-col-input", Input)
        old_name = col_sel.value
        new_name = rename_input.value.strip()
        if old_name is Select.BLANK:
            self.notify("Select a column to rename", severity="warning")
            return
        if not new_name:
            self.notify("Enter a new column name", severity="warning")
            return
        if new_name in self.current_df.columns:
            self.notify(f"Column '{new_name}' already exists!", severity="warning")
            return
        self.current_df = self.current_df.rename(columns={str(old_name): new_name})
        rename_input.value = ""
        self._save_current()
        self._load_file(self.current_file)
        self.notify(f"Renamed '{old_name}' to '{new_name}'")

    @on(Input.Changed, "#search-input")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._apply_search(event.value.strip())

    @on(DataTable.CellSelected, "#data-table")
    def _on_data_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Allow inline editing of a cell — prompt for new value."""
        if self.current_df is None or self.current_file is None:
            return
        row_idx = event.coordinate.row
        col_idx = event.coordinate.column
        if 0 <= row_idx < len(self.current_df) and 0 <= col_idx < len(self.current_df.columns):
            col_name = self.current_df.columns[col_idx]
            current_val = str(self.current_df.iloc[row_idx, col_idx])
            self.query_one("#row-status", Static).update(
                f"Row {row_idx + 1} / {len(self.current_df)}  \u2502  Column: {col_name}  \u2502  Value: {current_val}"
            )

    @on(Button.Pressed, "#plot-btn")
    def _on_plot(self) -> None:
        if self.current_df is None or self.current_file is None:
            self.notify("Open a CSV file first!", severity="warning")
            return
        x_sel: Select = self.query_one("#x-select", Select)
        y_sel: Select = self.query_one("#y-select", Select)
        chart_sel: Select = self.query_one("#chart-select", Select)
        x_col, y_col, chart_type = x_sel.value, y_sel.value, chart_sel.value
        if x_col is Select.BLANK or y_col is Select.BLANK:
            self.notify("Select both X and Y columns!", severity="warning")
            return
        if chart_type is Select.BLANK:
            chart_type = "line"
        self._generate_plot(str(x_col), str(y_col), str(chart_type))

    # ── plotting ─────────────────────────────────────────────────────────

    def _generate_plot(self, x_col: str, y_col: str, chart_type: str) -> None:
        df = self.current_df
        if df is None or df.empty:
            self.notify("No data to plot!", severity="warning")
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        x = df[x_col]
        y = pd.to_numeric(df[y_col], errors="coerce")
        title = f"{self.current_file.name} — {y_col} vs {x_col}"
        if chart_type == "line":
            ax.plot(x, y, marker="o", linewidth=2)
        elif chart_type == "bar":
            ax.bar(x, y)
        elif chart_type == "scatter":
            ax.scatter(x, y, s=60)
        elif chart_type == "area":
            ax.fill_between(range(len(x)), y, alpha=0.4)
            ax.plot(range(len(x)), y, linewidth=2)
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x, rotation=45, ha="right")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="csv_plot_") as tmp:
            fig.savefig(tmp.name, dpi=150)
            plt.close(fig)
            self._open_image(tmp.name)
            self.notify(f"Plot saved to {tmp.name}")
            self.query_one("#plot-status", Static).update(
                f"Last plot: {y_col} vs {x_col} ({chart_type})\nSaved to: {tmp.name}"
            )

    @staticmethod
    def _open_image(path: str) -> None:
        if os.name == "nt":
            os.startfile(path)
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── input shortcuts ──────────────────────────────────────────────────

    @on(Input.Submitted, "#name-input")
    def _on_name_submit(self) -> None:
        self._on_create()

    @on(Input.Submitted, "#columns-input")
    def _on_columns_submit(self) -> None:
        self._on_create()

    @on(Input.Submitted, "#new-col-input")
    def _on_col_submit(self) -> None:
        self.query_one("#add-col-btn", Button).press()

    @on(Input.Submitted, "#rename-col-input")
    def _on_rename_submit(self) -> None:
        self.query_one("#rename-col-btn", Button).press()

    def action_toggle_theme(self) -> None:
        themes = list(self.available_themes.keys())
        current = themes.index(self.theme) if self.theme in themes else 0
        self.theme = themes[(current + 1) % len(themes)]
        cfg = _load_config()
        cfg["theme"] = self.theme
        _save_config(cfg)
        self.notify(f"Theme: {self.theme}")

    # -- git sync ----------------------------------------------------------

    def action_git_sync(self) -> None:
        self.notify("Syncing with git...")
        self.run_worker(self._git_sync_worker(), exclusive=True)

    async def _git_sync_worker(self) -> None:
        import asyncio
        data_path = str(DATA_DIR.parent)  # data/ folder
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "pull", "--rebase",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, err = await proc.communicate()
            if proc.returncode != 0 and err:
                self.notify(f"Pull failed: {err.decode().strip()}", severity="error")
                return

            proc = await asyncio.create_subprocess_exec(
                "git", "add", data_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "--cached", "--quiet",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode == 0:
                self.notify("Already up to date")
                self._refresh_file_list()
                return

            proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", "csv-manager: sync data",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, err = await proc.communicate()
            if proc.returncode != 0:
                self.notify(f"Commit failed: {err.decode().strip()}", severity="error")
                return

            proc = await asyncio.create_subprocess_exec(
                "git", "push",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, err = await proc.communicate()
            if proc.returncode != 0:
                self.notify(f"Push failed: {err.decode().strip()}", severity="error")
                return

            self.notify("Data synced to git")
            self._refresh_file_list()
        except FileNotFoundError:
            self.notify("Git not found - is git installed?", severity="error")
        except Exception as exc:
            self.notify(str(exc), severity="error")


if __name__ == "__main__":
    CSVManagerApp().run()
