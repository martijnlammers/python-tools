from textual.app import App, ComposeResult
from textual.widgets import Button, Header, Footer, Input, Static, DataTable
from textual.containers import Horizontal, ScrollableContainer
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile
import os
import subprocess

DATA_DIR = Path("data/csv")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def sanitize_id(name: str) -> str:
    """Convert filename to a safe Textual widget ID."""
    sanitized = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    if sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized

class CSVManagerApp(App):
    """Terminal CSV manager with Create, Open, Delete, Plot buttons."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("CSV Manager", id="title")

        # Input + Create CSV button
        with Horizontal():
            self.csv_name_input = Input(placeholder="New CSV name (without .csv)")
            yield self.csv_name_input
            yield Button("Create CSV", id="create_csv")

        # Scrollable container for CSV list
        self.csv_list_container = ScrollableContainer()
        yield self.csv_list_container

        # Table to show opened CSV
        self.csv_table = DataTable(zebra_stripes=True)
        yield self.csv_table

        yield Footer()

    def on_mount(self):
        self.refresh_csv_list()
        self.set_focus(self.csv_name_input)

    # ---------------- CSV List ----------------
    def refresh_csv_list(self):
        """Rebuild the CSV list with buttons."""
        # Remove all existing rows
        for child in list(self.csv_list_container.children):
            self.csv_list_container.remove(child)

        for csv_file in sorted(DATA_DIR.glob("*.csv")):
            safe_name = sanitize_id(csv_file.name)
            row_container = Horizontal(id=f"row_{safe_name}")
            # Mount the container first to attach it
            self.csv_list_container.mount(row_container)
            # Now mount the child widgets
            row_container.mount(Static(csv_file.name, expand=True))
            row_container.mount(Button("Open", id=f"open_{safe_name}"))
            row_container.mount(Button("Plot", id=f"plot_{safe_name}"))
            row_container.mount(Button("Delete", id=f"delete_{safe_name}"))

    # ---------------- CSV Creation ----------------
    def create_csv(self):
        name = self.csv_name_input.value.strip()
        if name:
            file_path = DATA_DIR / f"{name}.csv"
            if file_path.exists():
                self.console.print(f"[red]CSV '{name}.csv' already exists![/red]")
            else:
                pd.DataFrame(columns=["Month", "Value", "Goal"]).to_csv(file_path, index=False)
                self.console.print(f"[green]Created CSV: {file_path.name}[/green]")
                self.csv_name_input.value = ""
                self.refresh_csv_list()
        else:
            self.console.print("[red]Please enter a valid CSV name[/red]")

    # ---------------- Button Handling ----------------
    def on_button_pressed(self, event: Button.Pressed):
        btn_id = event.button.id
        if btn_id == "create_csv":
            self.create_csv()
        elif btn_id.startswith("delete_"):
            filename = event.button.id.replace("delete_", "")
            self.delete_csv(filename)
        elif btn_id.startswith("open_"):
            filename = event.button.id.replace("open_", "")
            self.open_csv(filename)
        elif btn_id.startswith("plot_"):
            filename = event.button.id.replace("plot_", "")
            self.plot_csv(filename)

    # ---------------- Delete CSV ----------------
    def delete_csv(self, safe_name: str):
        for file in DATA_DIR.glob("*.csv"):
            if sanitize_id(file.name) == safe_name:
                file.unlink()
                self.console.print(f"[yellow]Deleted CSV: {file.name}[/yellow]")
                self.refresh_csv_list()
                self.csv_table.clear()
                return

    # ---------------- Open CSV ----------------
    def open_csv(self, safe_name: str):
        for file in DATA_DIR.glob("*.csv"):
            if sanitize_id(file.name) == safe_name:
                df = pd.read_csv(file)
                self.csv_table.clear(columns=True)
                self.csv_table.add_columns(*df.columns)
                for _, row in df.iterrows():
                    self.csv_table.add_row(*[str(cell) for cell in row.values])
                self.console.print(f"[green]Opened CSV: {file.name}[/green]")
                return

    # ---------------- Plot CSV ----------------
    def plot_csv(self, safe_name: str):
        for file in DATA_DIR.glob("*.csv"):
            if sanitize_id(file.name) == safe_name:
                df = pd.read_csv(file)
                if "Month" not in df.columns or "Value" not in df.columns or "Goal" not in df.columns:
                    self.console.print("[red]CSV must have Month, Value, Goal columns to plot[/red]")
                    return
                plt.figure(figsize=(6, 4))
                plt.plot(df["Month"], df["Value"], marker="o", label="Value")
                plt.plot(df["Month"], df["Goal"], linestyle="--", label="Goal")
                plt.xlabel("Month")
                plt.ylabel("Value")
                plt.title(file.name)
                plt.legend()
                plt.tight_layout()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    plt.savefig(tmp.name)
                    plt.close()
                    if os.name == "nt":
                        os.startfile(tmp.name)
                    elif os.name == "posix":
                        subprocess.run(["xdg-open", tmp.name])
                    else:
                        self.console.print(f"[yellow]Plot saved to {tmp.name}[/yellow]")
                return

    # ---------------- Input Enter ----------------
    def on_input_submitted(self, event):
        if event.input == self.csv_name_input:
            self.create_csv()


if __name__ == "__main__":
    CSVManagerApp().run()