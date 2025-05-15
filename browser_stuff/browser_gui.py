# browser_gui.py – Tk control panel for browserlib
#
# v8  (2025-04-24)
# • Element list placed in a Listbox + Scrollbar container.
# • refresh() preserves current y-scroll position, so autorefresh doesn’t
#   jerk the view back to the top while you’re browsing the list.
#
import asyncio, threading, tkinter as tk, tkinter.ttk as ttk, argparse, sys
from typing import Dict, List, Tuple
from core import Browser

REFRESH_MS = 1000


class ControlGUI:
    # ──────────────────────────── bootstrap ────────────────────────────── #
    def __init__(self, url: str, start_overlay: bool):
        # background event loop + browser
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.browser = Browser()
        asyncio.run_coroutine_threadsafe(self._async_setup(url), self.loop)

        # Tk window
        self.overlay_on = start_overlay
        root = self.root = tk.Tk()
        root.title("browserlib control panel")

        # ── tab selector row ─────────────────────────────────────────────── #
        self._build_tab_row(root)

        # ── scrollable element list ─────────────────────────────────────── #
        self._build_listbox(root)

        # ── action rows (click/type/scroll etc.) ────────────────────────── #
        self._build_action_rows(root)

        # window chrome
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(700, self._update_tabs)
        root.after(700, self.refresh)
        root.after(REFRESH_MS, self._auto_refresh)
        root.mainloop()

    # ────────────────── widget-construction helpers ─────────────────────── #
    def _build_tab_row(self, root: tk.Tk):
        tab_row = ttk.Frame(root); tab_row.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(tab_row, text="Tab:").pack(side="left")

        self.tab_var = tk.StringVar()
        self.tab_combo = ttk.Combobox(
            tab_row, textvariable=self.tab_var, state="readonly", width=50
        )
        self.tab_combo.pack(side="left", expand=True, fill="x")
        self.tab_combo.bind("<<ComboboxSelected>>", self._tab_changed)

        ttk.Button(tab_row, text="↻", width=3,
                   command=self._update_tabs).pack(side="left", padx=(4, 0))

    def _build_listbox(self, root: tk.Tk):
        list_frame = ttk.Frame(root); list_frame.pack(
            fill="both", expand=True, padx=6, pady=6
        )

        self.listbox = tk.Listbox(
            list_frame, width=72, font=("Consolas", 10), exportselection=False
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<Double-Button-1>", self._click_selected)

        # mouse-wheel scrolling
        self.listbox.bind(
            "<MouseWheel>",
            lambda e: self.listbox.yview_scroll(int(-e.delta / 120), "units")
        )
        self.listbox.bind(
            "<Button-4>",  # Linux wheel-up
            lambda e: self.listbox.yview_scroll(-1, "units")
        )
        self.listbox.bind(
            "<Button-5>",  # Linux wheel-down
            lambda e: self.listbox.yview_scroll(1, "units")
        )

        sb = ttk.Scrollbar(list_frame, orient="vertical",
                           command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=sb.set)

    def _build_action_rows(self, root: tk.Tk):
        # ----- Click / type row ------------------------------------------- #
        entry_row = ttk.Frame(root); entry_row.pack(fill="x", padx=6)

        ttk.Label(entry_row, text="ID:").pack(side="left")
        self.id_entry = ttk.Entry(entry_row, width=6)
        self.id_entry.pack(side="left")
        self.id_entry.bind("<Return>", self._click_typed)

        ttk.Button(entry_row, text="Click",
                   command=self._click_any).pack(side="left", padx=(4, 10))

        ttk.Label(entry_row, text="Text:").pack(side="left")
        self.text_entry = ttk.Entry(entry_row, width=28)
        self.text_entry.pack(side="left")
        self.text_entry.bind("<Return>", self._type_any)

        ttk.Button(entry_row, text="Type",
                   command=self._type_any).pack(side="left", padx=(4, 10))

        ttk.Button(entry_row, text="Refresh",
                   command=self.refresh).pack(side="left")

        self.ov_btn = ttk.Button(
            entry_row,
            text="Overlay off" if self.overlay_on else "Overlay on",
            command=self.toggle_overlay,
        )
        self.ov_btn.pack(side="left", padx=4)

        # ----- Scroll row -------------------------------------------------- #
        scroll_row = ttk.Frame(root); scroll_row.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(scroll_row, text="Pixels:").pack(side="left")
        self.scroll_entry = ttk.Entry(scroll_row, width=6)
        self.scroll_entry.pack(side="left")
        self.scroll_entry.insert(0, "400")
        self.scroll_entry.bind("<Return>", self._scroll_down)

        ttk.Button(scroll_row, text="Scroll ▲",
                   command=self._scroll_up).pack(side="left", padx=4)
        ttk.Button(scroll_row, text="Scroll ▼",
                   command=self._scroll_down).pack(side="left", padx=4)

        # ----- New-tab row -------------------------------------------------- #
        new_row = ttk.Frame(root); new_row.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(new_row, text="URL:").pack(side="left")
        self.newtab_entry = ttk.Entry(new_row, width=36)
        self.newtab_entry.pack(side="left", expand=True, fill="x")
        self.newtab_entry.bind("<Return>", self._open_tab)

        ttk.Button(new_row, text="New Tab",
                   command=self._open_tab).pack(side="left", padx=4)

    # ───────────────────── background-thread helpers ────────────────────── #
    async def _async_setup(self, url: str):
        await self.browser.start()
        await self.browser.goto(url)
        if self.overlay_on:
            await self.browser.enable_overlay()

    def _run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    # ────────────────────────── tab control ─────────────────────────────── #
    def _update_tabs(self):
        fut = self._run_async(self.browser.list_tabs())
        try:
            tabs: List[Tuple[int, str]] = fut.result(timeout=2)
        except Exception:
            return
        labels = [f"{i}: {label}" for i, label in tabs]
        cur = self.tab_combo.get()
        self.tab_combo["values"] = labels
        if cur in labels:
            self.tab_combo.set(cur)
        elif labels:
            self.tab_combo.current(0)

    def _tab_changed(self, *_):
        sel = self.tab_combo.get()
        if not sel:
            return
        idx = int(sel.split(":")[0])
        self._run_async(self.browser.activate_tab(idx))
        self.root.after(500, self.refresh)

    # ───────────────────────── element list ─────────────────────────────── #
    def refresh(self):
        # remember scroll position
        top_frac = self.listbox.yview()[0] if self.listbox.size() else 0.0

        fut = self._run_async(self.browser.refresh_map())
        try:
            elements: Dict[int, Dict] = fut.result()
        except Exception as e:
            print("refresh_map failed:", e)
            return

        self.listbox.delete(0, tk.END)
        for i, meta in elements.items():
            label = meta["label"] or "(no label)"
            self.listbox.insert(
                tk.END, f"{i:>3d}: {meta['tag']:<8} {meta['kind']:<6} {label}"
            )

        # restore previous scroll location
        self.listbox.yview_moveto(top_frac)

    # ───────────────────── click / type / scroll ────────────────────────── #
    def _click_selected(self, *_):
        sel = self.listbox.curselection()
        if sel:
            elem_id = int(self.listbox.get(sel[0]).split(":")[0])
            self._run_async(self.browser.click(elem_id))

    def _click_typed(self, *_): self._click_any()

    def _click_any(self):
        txt = self.id_entry.get().strip()
        if txt.isdigit():
            self._run_async(self.browser.click(int(txt)))
            self.id_entry.delete(0, tk.END)
        else:
            self._click_selected()

    def _type_any(self, *_):
        id_txt = self.id_entry.get().strip()
        text = self.text_entry.get()
        if not text:
            return
        if id_txt.isdigit():
            elem_id = int(id_txt)
        else:
            sel = self.listbox.curselection()
            if not sel:
                return
            elem_id = int(self.listbox.get(sel[0]).split(":")[0])
        self._run_async(self.browser.input_text(elem_id, text))
        self.text_entry.delete(0, tk.END)

    def _get_pixels(self) -> int | None:
        txt = self.scroll_entry.get().strip()
        return int(txt) if txt.lstrip("-").isdigit() else None

    def _scroll_up(self, *_):
        px = self._get_pixels()
        if px is None:
            return
        id_txt = self.id_entry.get().strip()
        if id_txt.isdigit():
            self._run_async(self.browser.scroll_element_up(int(id_txt), px))
        else:
            self._run_async(self.browser.scroll_up(px))

    def _scroll_down(self, *_):
        px = self._get_pixels()
        if px is None:
            return
        id_txt = self.id_entry.get().strip()
        if id_txt.isdigit():
            self._run_async(self.browser.scroll_element_down(int(id_txt), px))
        else:
            self._run_async(self.browser.scroll_down(px))

    # ───────────────────────── overlay toggle ───────────────────────────── #
    def toggle_overlay(self):
        self.overlay_on = not self.overlay_on
        if self.overlay_on:
            self._run_async(self.browser.enable_overlay())
            self.ov_btn.config(text="Overlay off")
        else:
            self._run_async(self.browser.disable_overlay())
            self.ov_btn.config(text="Overlay on")

    # ───────────────────── new-tab helper --------------------------------- #
    def _open_tab(self, *_):
        url = self.newtab_entry.get().strip()
        if not url:
            return
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        self._run_async(self.browser.create_new_tab(url))
        self.newtab_entry.delete(0, tk.END)
        self.root.after(1200, self._update_tabs)
        self.root.after(1200, self.refresh)

    # ───────────────────────── misc housekeeping ────────────────────────── #
    def _auto_refresh(self):
        self.refresh()
        self.root.after(REFRESH_MS, self._auto_refresh)

    def _on_close(self):
        self._run_async(self.browser.close())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Tk control panel for browserlib")
    ap.add_argument("url", nargs="?", default="https://google.com")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()
    ControlGUI(args.url, start_overlay=not args.no_overlay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
