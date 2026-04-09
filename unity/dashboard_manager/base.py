"""
Base abstract class for DashboardManager.

Defines the public contract for creating and managing dashboard tiles and
layouts. All docstrings are defined here and inherited by concrete
implementations via ``@functools.wraps``.

IMPORTANT: Do not duplicate docstrings in concrete implementations.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

from unity.common.state_managers import BaseStateManager
from unity.dashboard_manager.types.tile import (
    DataBinding,
    TileRecord,
    TileResult,
)
from unity.dashboard_manager.types.dashboard import (
    DashboardRecord,
    DashboardResult,
    TilePosition,
)


class BaseDashboardManager(BaseStateManager):
    """
    Public contract for the dashboard and tile management layer.

    DashboardManager provides synchronous primitives for creating, retrieving,
    updating, and deleting HTML visualization tiles and composed dashboard
    layouts.  It is the single source of truth for:

    - **Tile CRUD**: create_tile, get_tile, update_tile, delete_tile, list_tiles
    - **Dashboard CRUD**: create_dashboard, get_dashboard, update_dashboard,
      delete_dashboard, list_dashboards

    Design Philosophy
    -----------------
    The actor generates arbitrary HTML visualizations using Python (Plotly,
    Bokeh, Matplotlib, custom HTML/CSS/JS, etc.) and passes the resulting
    HTML string to ``create_tile()``.  This decouples visualization
    generation from any specific frontend library and replaces the old
    ``plot()`` / ``table_view()`` primitives on DataManager.  The actor
    should **always** use DashboardManager for visualizations.

    Tiles support two data modes:

    1. **Baked-in data** -- HTML embeds all data directly (e.g., Plotly's
       ``fig.to_html(include_plotlyjs='cdn')``).  Best for small-to-medium
       datasets or one-time snapshots.

    2. **Live data bridge** -- HTML calls ``UnifyData`` methods
       (``filter``, ``reduce``, ``join``, ``joinReduce``) that fetch or
       aggregate fresh data at render time via postMessage.  Best for large
       or frequently updated datasets.  Declare ``data_bindings`` to
       specify which Unify contexts and operations the tile uses.

    Dashboards compose multiple tiles into a responsive 12-column grid layout.
    Each tile is placed using ``TilePosition(tile_token, x, y, w, h)`` where
    ``x`` is the column offset (0--11), ``w`` is the width in columns (1--12),
    and ``y``/``h`` control vertical placement in row units.

    Tokens and URLs
    ---------------
    Every tile and dashboard is identified by a 12-character URL-safe token
    generated at creation time.  The token is embedded in the shareable URL
    (e.g., ``/tile/view/{token}`` or ``/dashboard/view/{token}``).  Tokens
    are stable and can be stored for later retrieval.

    Tile ↔ Dashboard Lifecycle
    --------------------------
    Tiles and dashboards have **independent lifecycles**:

    - Deleting a dashboard does **not** delete its tiles -- they remain
      accessible via their tokens and can be reused in other dashboards.
    - Deleting a tile does **not** remove it from dashboard layouts -- the
      dashboard layout JSON will reference a now-missing tile.  Clean up
      stale references by updating the dashboard layout.

    No Tool Loops
    -------------
    DashboardManager exposes pure primitives with no ask/update tool loops.
    It follows the same stateless API pattern as DataManager.

    Docstring Requirements for Subclass Methods
    -------------------------------------------
    All public methods (primitives) MUST include comprehensive docstrings with:

    1. **One-line summary** -- What the method does
    2. **Extended description** -- When to use, contrasted with similar methods,
       design reasoning
    3. **Parameters section** -- EVERY parameter with:
       - Type annotation
       - Detailed description of expected values
       - Default behavior when optional
       - Examples of valid values
    4. **Returns section** -- Return type AND structure details (what fields
       it contains, when fields are populated vs ``None``)
    5. **Raises section** -- Exceptions that may be raised
    6. **Usage Examples** -- MULTIPLE concrete code examples showing common
       patterns (baked-in data, live data bridge, Plotly, custom HTML, etc.)
    7. **Anti-patterns section** -- What NOT to do and why
    8. **Notes section** -- Invariants, edge cases, additional context
    9. **See Also section** -- Cross-references to related methods

    This is CRITICAL because:
    - Actor/FunctionManager reads docstrings to understand primitive usage
    - LLMs compose primitives based on docstring content
    - No external documentation -- docstrings ARE the documentation
    """

    _as_caller_description: str = (
        "a DashboardManager, creating visualization tiles and dashboard layouts"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Tiles
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def create_tile(
        self,
        html: str,
        *,
        title: str,
        description: Optional[str] = None,
        data_bindings: Optional[List[DataBinding]] = None,
    ) -> TileResult:
        """
        Create a new visualization tile from an HTML string.

        This is the primary entry point for all visualizations.  The actor
        generates arbitrary HTML in Python -- Plotly charts, Bokeh plots,
        Matplotlib figures exported to HTML, custom D3/CSS/JS dashboards,
        KPI cards, rich tables -- and passes the resulting HTML string here.
        The tile is stored in the ``Dashboards/Tiles`` Unify context and
        assigned a 12-character URL-safe token for shareable access.

        This method replaces the old ``primitives.data.plot()`` and
        ``primitives.data.table_view()`` primitives.  The actor should
        **always** use ``create_tile()`` for any visualization output.

        Tiles support two data modes controlled by whether ``data_bindings``
        is provided:

        - **Baked-in data** (no ``data_bindings``): The HTML embeds all
          data directly.  Best for small-to-medium datasets, one-time
          snapshots, or static charts where the data won't change.
        - **Live data bridge** (with ``data_bindings``): The HTML calls
          ``UnifyData`` methods that fetch/aggregate fresh data at render
          time via postMessage.  Best for large datasets, frequently
          updated data, or dashboards that should always reflect the
          latest state.

        Parameters
        ----------
        html : str
            Self-contained HTML content for the tile.  Must be a complete
            HTML document or fragment that renders correctly in an iframe.
            The content is stored as-is in the ``html_content`` field of the
            tile record.

            **Baked-in data examples**:

            - Plotly: ``fig.to_html(include_plotlyjs='cdn', full_html=True)``
            - Matplotlib: Export to SVG/PNG and embed in an ``<img>`` tag,
              or use ``mpld3`` for interactive HTML.
            - Bokeh: ``bokeh.embed.file_html(plot, CDN, "title")``
            - Custom: Any valid HTML with inline CSS/JS.

            **Live data bridge examples**::

                <!-- Simple filter query -->
                UnifyData.filter({
                  context: "Data/Sales/Monthly",
                  filter: "year == 2025",
                  columns: ["month", "revenue"],
                }).then(rows => renderChart(rows));

                <!-- Aggregation -->
                UnifyData.reduce({
                  context: "Data/Sales/Monthly",
                  metric: "sum",
                  columns: "revenue",
                  group_by: ["region"],
                }).then(result => renderKPI(result));

                <!-- Cross-table join -->
                UnifyData.join({
                  tables: ["Data/Orders", "Data/Customers"],
                  join_expr: "Data/Orders.customer_id == Data/Customers.id",
                  select: {"Data/Orders.amount": "amount",
                           "Data/Customers.name": "customer"},
                }).then(rows => renderTable(rows));

            When using CDN-hosted libraries, include them via ``<script>``
            tags pointing to CDN URLs -- the iframe has no access to locally
            installed Python packages.

        title : str
            Human-readable title displayed in the tile header and used for
            discovery via ``list_tiles()``.  Should be concise but descriptive
            enough to distinguish the tile from others.

            Examples: ``"Revenue by Category"``, ``"Q4 KPI Summary"``,
            ``"Arrears Heatmap by Region"``

        description : str | None, default ``None``
            Longer description of what the tile shows, its data source, or
            how to interpret it.  Useful for discoverability when browsing
            tiles via ``list_tiles()``.  When ``None``, only the title is
            stored.

        data_bindings : list[DataBinding] | None, default ``None``
            Declared data sources for live-data tiles.  ``DataBinding`` is a
            discriminated union (on the ``operation`` field) of four types,
            each mapping to a ``UnifyData`` JS method and a ``DataManager``
            primitive:

            **FilterBinding** (``operation="filter"``, default):
              Single-context row fetch.  JS: ``UnifyData.filter()``.
              Validated via ``DataManager.filter(limit=5)``.

              Fields: ``context`` (required), ``alias``, ``filter``,
              ``columns``, ``exclude_columns``, ``order_by``, ``descending``,
              ``limit``, ``offset``, ``group_by``.

            **ReduceBinding** (``operation="reduce"``):
              Single-context aggregation.  JS: ``UnifyData.reduce()``.
              Validated via ``DataManager.reduce()``.

              Fields: ``context`` (required), ``metric`` (required),
              ``columns`` (required), ``alias``, ``filter``, ``group_by``,
              ``result_where``.

            **JoinBinding** (``operation="join"``):
              Cross-context join returning rows.  JS: ``UnifyData.join()``.
              Validated via ``DataManager.filter_join(result_limit=5)``.

              Fields: ``tables`` (required, exactly 2 context paths),
              ``join_expr`` (required), ``select`` (required),
              ``alias``, ``mode``, ``left_where``, ``right_where``,
              ``result_where``, ``result_limit``, ``result_offset``.

            **JoinReduceBinding** (``operation="join_reduce"``):
              Cross-context join + aggregation.  JS: ``UnifyData.joinReduce()``.
              Validated via ``DataManager.reduce_join()``.

              Fields: ``tables`` (required), ``join_expr`` (required),
              ``select`` (required), ``metric`` (required),
              ``columns`` (required), ``alias``, ``mode``, ``left_where``,
              ``right_where``, ``group_by``, ``result_where``.

            **Auto-validation**: When ``data_bindings`` is provided,
            ``create_tile`` automatically dry-runs each binding through
            the corresponding ``DataManager`` method before storing the
            tile.  If any binding references a nonexistent context,
            misspelled column, invalid expression, or incompatible metric,
            the tile is **not** stored and ``TileResult.error`` reports the
            problem.

            When provided:

            - ``has_data_bindings`` is set to ``True`` on the tile record.
            - ``data_binding_contexts`` stores the comma-separated context
              paths (including both tables for join bindings).
            - The data bridge is activated so ``UnifyData`` calls in the
              HTML resolve at render time.

            When ``None`` (the default), the tile operates in baked-in data
            mode and no bridge script is injected.

            Example -- filter binding::

                data_bindings=[
                    FilterBinding(
                        context="Data/Sales/Monthly",
                        columns=["month", "revenue"],
                        order_by="month",
                    ),
                ]

            Example -- reduce binding (live KPI card)::

                data_bindings=[
                    ReduceBinding(
                        context="Data/Sales/Monthly",
                        metric="sum",
                        columns="revenue",
                        group_by=["region"],
                    ),
                ]

            Example -- join binding (cross-table chart)::

                data_bindings=[
                    JoinBinding(
                        tables=["Data/Orders", "Data/Customers"],
                        join_expr="Data/Orders.cust_id == Data/Customers.id",
                        select={
                            "Data/Orders.amount": "amount",
                            "Data/Customers.name": "customer",
                        },
                        result_limit=500,
                    ),
                ]

            Example -- join-reduce binding (cross-table KPI)::

                data_bindings=[
                    JoinReduceBinding(
                        tables=["Data/Orders", "Data/Products"],
                        join_expr="Data/Orders.product_id == Data/Products.id",
                        select={
                            "Data/Orders.amount": "amount",
                            "Data/Products.category": "category",
                        },
                        metric="sum",
                        columns="amount",
                        group_by=["category"],
                    ),
                ]

        Returns
        -------
        TileResult
            Result object with the following fields:

            - ``url`` (str | None): Shareable URL for viewing the tile.
              ``None`` on failure.
            - ``token`` (str | None): The 12-character URL-safe token
              assigned to this tile.  Use this to reference the tile in
              ``get_tile()``, ``update_tile()``, ``delete_tile()``, and
              ``TilePosition.tile_token`` when composing dashboards.
            - ``title`` (str | None): The title that was stored.
            - ``error`` (str | None): Error message if the operation failed.
              ``None`` on success.
            - ``succeeded`` (bool, property): ``True`` if ``url`` is not
              ``None`` and ``error`` is ``None``.  Always check this before
              using the URL or token.

        Raises
        ------
        ValueError
            If ``html`` is empty or ``title`` is empty.

        Usage Examples
        --------------
        # Baked-in data: Plotly bar chart
        import plotly.express as px
        fig = px.bar(df, x="category", y="revenue", title="Revenue")
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        result = primitives.dashboards.create_tile(
            html, title="Revenue by Category"
        )
        print(result.url)

        # Live filter: Chart that fetches fresh rows on render
        html = '''
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <div id="chart"></div>
        <script>
          UnifyData.filter({
            context: "Data/Sales/Monthly",
            filter: "year == 2025",
            columns: ["month", "revenue"],
            order_by: "month",
          }).then(rows => {
            Plotly.newPlot("chart", [{
              x: rows.map(r => r.month),
              y: rows.map(r => r.revenue),
              type: "bar",
            }]);
          });
        </script>
        '''
        result = primitives.dashboards.create_tile(
            html,
            title="Monthly Revenue (Live)",
            data_bindings=[
                FilterBinding(
                    context="Data/Sales/Monthly",
                    filter="year == 2025",
                    columns=["month", "revenue"],
                    order_by="month",
                ),
            ],
        )

        # Live reduce: KPI card showing aggregated total
        html = '''
        <div id="kpi" style="font-family:sans-serif;padding:20px;"></div>
        <script>
          UnifyData.reduce({
            context: "Data/Sales/Monthly",
            metric: "sum",
            columns: "revenue",
          }).then(val => {
            document.getElementById("kpi").innerHTML =
              `<h2>Total Revenue</h2><p style="font-size:48px;">$${val.toLocaleString()}</p>`;
          });
        </script>
        '''
        result = primitives.dashboards.create_tile(
            html,
            title="Revenue KPI (Live)",
            data_bindings=[
                ReduceBinding(
                    context="Data/Sales/Monthly",
                    metric="sum",
                    columns="revenue",
                ),
            ],
        )

        # Live join: Table from two joined contexts
        html = '''
        <div id="tbl"></div>
        <script>
          UnifyData.join({
            tables: ["Data/Orders", "Data/Customers"],
            join_expr: "Data/Orders.cust_id == Data/Customers.id",
            select: {"Data/Orders.amount": "amount",
                     "Data/Customers.name": "customer"},
            result_limit: 100,
          }).then(rows => {
            let h = "<table><tr><th>Customer</th><th>Amount</th></tr>";
            rows.forEach(r => { h += `<tr><td>${r.customer}</td><td>${r.amount}</td></tr>`; });
            document.getElementById("tbl").innerHTML = h + "</table>";
          });
        </script>
        '''
        result = primitives.dashboards.create_tile(
            html,
            title="Order Details (Live Join)",
            data_bindings=[
                JoinBinding(
                    tables=["Data/Orders", "Data/Customers"],
                    join_expr="Data/Orders.cust_id == Data/Customers.id",
                    select={
                        "Data/Orders.amount": "amount",
                        "Data/Customers.name": "customer",
                    },
                    result_limit=100,
                ),
            ],
        )

        # Live join-reduce: KPI aggregated across joined tables
        html = '''
        <div id="kpi"></div>
        <script>
          UnifyData.joinReduce({
            tables: ["Data/Orders", "Data/Products"],
            join_expr: "Data/Orders.product_id == Data/Products.id",
            select: {"Data/Orders.amount": "amount",
                     "Data/Products.category": "category"},
            metric: "sum",
            columns: "amount",
            group_by: ["category"],
          }).then(result => {
            let h = "<ul>";
            for (const [cat, val] of Object.entries(result))
              h += `<li>${cat}: $${val.toLocaleString()}</li>`;
            document.getElementById("kpi").innerHTML = h + "</ul>";
          });
        </script>
        '''
        result = primitives.dashboards.create_tile(
            html,
            title="Revenue by Category (Live Join-Reduce)",
            data_bindings=[
                JoinReduceBinding(
                    tables=["Data/Orders", "Data/Products"],
                    join_expr="Data/Orders.product_id == Data/Products.id",
                    select={
                        "Data/Orders.amount": "amount",
                        "Data/Products.category": "category",
                    },
                    metric="sum",
                    columns="amount",
                    group_by=["category"],
                ),
            ],
        )

        UnifyData JS API Reference
        --------------------------
        The live data bridge exposes four methods on ``window.UnifyData``,
        each mapping to a ``DataManager`` primitive.  Parameter names
        mirror the Python ``primitives.data.*`` API exactly.

        **UnifyData.filter(opts)** -> ``Promise<Array<Object>>``
          Fetch rows from a single context.

          Required: ``context`` (string).
          Optional: ``filter``, ``columns``, ``exclude_columns``,
          ``order_by``, ``descending``, ``limit``, ``offset``,
          ``group_by``, ``sorting``, ``column_context``, ``randomize``.

        **UnifyData.reduce(opts)** -> ``Promise<Any>``
          Compute an aggregate metric over a single context.

          Required: ``context`` (string), ``metric`` (string),
          ``columns`` (string | string[]).
          Optional: ``filter``, ``group_by``, ``result_where``.

        **UnifyData.join(opts)** -> ``Promise<Array<Object>>``
          Join two contexts and return rows.

          Required: ``tables`` (string[2]), ``join_expr`` (string),
          ``select`` (object).
          Optional: ``mode``, ``left_where``, ``right_where``,
          ``result_where``, ``result_limit``, ``result_offset``.

        **UnifyData.joinReduce(opts)** -> ``Promise<Any>``
          Join two contexts and aggregate the result.

          Required: ``tables`` (string[2]), ``join_expr`` (string),
          ``select`` (object), ``metric`` (string),
          ``columns`` (string | string[]).
          Optional: ``mode``, ``left_where``, ``right_where``,
          ``group_by``, ``result_where``.

        Verification-First Pattern (automatic for live data)
        ----------------------------------------------------
        ``create_tile`` **automatically** validates each binding through
        the corresponding ``DataManager`` method before the tile is
        stored:

        - ``FilterBinding``     -> ``DataManager.filter(limit=5)``
        - ``ReduceBinding``     -> ``DataManager.reduce(...)``
        - ``JoinBinding``       -> ``DataManager.filter_join(result_limit=5)``
        - ``JoinReduceBinding`` -> ``DataManager.reduce_join(...)``

        If any binding fails, the tile is **not** stored and
        ``TileResult.error`` explains the problem.

        Data mode decision framework
        ----------------------------
        BAKED-IN data (embed in HTML) when:
          - Dataset is small-to-medium (< 10k rows)
          - Data is a one-time snapshot or static report
          - Maximum portability desired (self-contained HTML)

        LIVE ``FilterBinding`` when:
          - Data is frequently updated, tile should reflect latest state
          - Dataset is large; fetch only what the tile needs
          - Simple row-level queries (filter, sort, paginate)

        LIVE ``ReduceBinding`` when:
          - Tile shows a KPI, total, average, or other aggregate
          - Server-side aggregation avoids transferring all rows

        LIVE ``JoinBinding`` when:
          - Tile needs data spanning two related contexts (star schema)
          - Example: orders joined with customers for a detail table

        LIVE ``JoinReduceBinding`` when:
          - Tile shows aggregated KPIs across joined contexts
          - Example: total revenue by product category from orders+products

        Anti-patterns
        -------------
        - WRONG: Using ``primitives.data.plot()`` or ``primitives.data.table_view()``.
          CORRECT: Generate HTML in Python and use ``create_tile()`` instead.
          The old plot/table_view primitives have been removed.

        - WRONG: Forgetting ``include_plotlyjs='cdn'`` when using Plotly.
          CORRECT: Always include ``include_plotlyjs='cdn'`` -- the iframe
          has no access to locally installed Python packages.

        - WRONG: Using ``UnifyData`` methods without declaring ``data_bindings``.
          CORRECT: Always pass ``data_bindings`` listing every operation the
          tile's JS code performs.  Without it, the bridge script is not
          injected and ``UnifyData`` will be undefined.

        - WRONG: Baking in data because the query needs joins or aggregation.
          CORRECT: Use ``JoinBinding`` / ``ReduceBinding`` /
          ``JoinReduceBinding`` so the tile fetches live cross-table or
          aggregated data at render time.

        - WRONG: Storing megabytes of data in baked-in HTML for live datasets.
          CORRECT: Use the live data bridge for large or frequently updated
          data.  Baked-in is best for snapshots under ~1 MB of HTML.

        - WRONG: Creating tiles without checking ``result.succeeded``.
          CORRECT: Always check ``result.succeeded`` before using the URL
          or token.

        - WRONG: Using positional arguments ``UnifyData.filter(context, filter)``.
          CORRECT: Always pass a single options object
          ``UnifyData.filter({context: "...", filter: "..."})``.

        - WRONG: Using internal backend param names in JS
          (``filter_expr``, ``from_fields``, ``exclude_fields``).
          CORRECT: Use the documented names (``filter``, ``columns``,
          ``exclude_columns``).  The mapping is handled automatically.

        Notes
        -----
        - The token is generated server-side and is guaranteed unique.
        - The URL is a shareable link, not a raw file URL.
        - Tiles are stored in the ``Dashboards/Tiles`` Unify context with
          auto-incrementing ``tile_id``.
        - The ``html_content`` is stored verbatim -- no sanitization or
          transformation is applied.
        - For Bokeh, use ``bokeh.embed.file_html(plot, CDN, "title")`` to
          get a self-contained HTML string.

        See Also
        --------
        get_tile : Retrieve a tile by token (includes full HTML).
        update_tile : Modify a tile's HTML or metadata.
        list_tiles : Discover existing tiles (without HTML content).
        create_dashboard : Compose tiles into a grid layout.
        """

    @abstractmethod
    def get_tile(self, token: str) -> Optional[TileRecord]:
        """
        Retrieve a tile by its token, including the full HTML content.

        Use this to fetch the complete tile record when you need the actual
        HTML content -- for example, to inspect a tile before updating it,
        to clone a tile's HTML for a variant, or to verify what was stored.

        This is the **only** method that returns ``html_content``.  The
        ``list_tiles()`` method deliberately excludes HTML to keep payloads
        small.  If you only need metadata (title, description, token,
        timestamps), use ``list_tiles()`` instead.

        Parameters
        ----------
        token : str
            The 12-character URL-safe token identifying the tile.  This is
            the value returned in ``TileResult.token`` from ``create_tile()``
            or ``update_tile()``.

            Example: ``"Ab3xK9mP2qLz"``

        Returns
        -------
        TileRecord | None
            The full tile record when found, or ``None`` if no tile exists
            with the given token.

            The ``TileRecord`` contains:

            - ``tile_id`` (int | None): Auto-incremented numeric identifier.
            - ``token`` (str): The 12-char URL-safe token.
            - ``title`` (str): Human-readable tile title.
            - ``description`` (str | None): Optional longer description.
            - ``html_content`` (str): The full HTML content of the tile.
            - ``has_data_bindings`` (bool): Whether the tile uses the live
              data bridge.
            - ``data_binding_contexts`` (str | None): Comma-separated Unify
              context paths if data bindings are declared.
            - ``created_at`` (str | None): ISO-8601 creation timestamp.
            - ``updated_at`` (str | None): ISO-8601 last-update timestamp.

        Usage Examples
        --------------
        # Retrieve a tile to inspect its HTML
        tile = primitives.dashboards.get_tile("Ab3xK9mP2qLz")
        if tile is not None:
            print(f"Title: {tile.title}")
            print(f"HTML length: {len(tile.html_content)} chars")
            print(f"Has live data: {tile.has_data_bindings}")

        # Clone a tile with modified HTML
        original = primitives.dashboards.get_tile(existing_token)
        if original is not None:
            new_html = original.html_content.replace("2024", "2025")
            result = primitives.dashboards.create_tile(
                new_html, title=f"{original.title} (2025 Update)"
            )

        # Check if a tile exists before referencing in a dashboard
        tile = primitives.dashboards.get_tile(token_from_config)
        if tile is None:
            print(f"Tile {token_from_config} not found -- skipping")

        Anti-patterns
        -------------
        - WRONG: Calling ``get_tile`` in a loop to list all tiles.
          CORRECT: Use ``list_tiles()`` to discover tiles efficiently.

        - WRONG: Using ``get_tile`` just to check existence.
          CORRECT: Use ``list_tiles(filter="token == '...'")`` for
          lightweight existence checks.

        Notes
        -----
        - Returns ``None`` (not an error) when the token does not match
          any tile.  Always check the return value before accessing fields.
        - The ``html_content`` may be large (megabytes for baked-in data
          tiles).  Only call ``get_tile`` when you actually need the HTML.

        See Also
        --------
        list_tiles : List tiles without HTML content (lightweight).
        update_tile : Modify a tile's HTML or metadata.
        create_tile : Create a new tile.
        """

    @abstractmethod
    def update_tile(
        self,
        token: str,
        *,
        html: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> TileResult:
        """
        Update an existing tile's content or metadata.

        Use this to refresh a tile's HTML (e.g., after regenerating a chart
        with new data), correct a title, or add a description -- without
        changing the tile's token or URL.  Only the fields you provide are
        updated; omitted fields retain their current values.  This makes
        ``update_tile`` ideal for incremental edits.

        The tile's token and URL remain stable across updates, so any
        dashboards referencing this tile will automatically show the new
        content on their next render.

        Parameters
        ----------
        token : str
            The 12-character URL-safe token of the tile to update.  Must
            reference an existing tile; use ``get_tile()`` or ``list_tiles()``
            to discover valid tokens.

            Example: ``"Ab3xK9mP2qLz"``

        html : str | None, default ``None``
            New HTML content to replace the tile's current ``html_content``.
            When ``None``, the existing HTML is preserved.

            Same requirements as ``create_tile()``'s ``html`` parameter:
            must be a self-contained HTML document or fragment that renders
            in an iframe.  Remember ``include_plotlyjs='cdn'`` for Plotly.

        title : str | None, default ``None``
            New title for the tile.  When ``None``, the existing title is
            preserved.

        description : str | None, default ``None``
            New description for the tile.  When ``None``, the existing
            description is preserved.

        Returns
        -------
        TileResult
            Result object with the same structure as ``create_tile()``:

            - ``url`` (str | None): The (unchanged) shareable URL.
            - ``token`` (str | None): The (unchanged) 12-char token.
            - ``title`` (str | None): The current title (updated if provided).
            - ``error`` (str | None): Error message if the update failed.
            - ``succeeded`` (bool, property): ``True`` on success.

        Raises
        ------
        ValueError
            If the token does not reference an existing tile.

        Usage Examples
        --------------
        # Refresh chart HTML with new data (keep title and description)
        fig = px.bar(updated_df, x="month", y="revenue")
        new_html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        result = primitives.dashboards.update_tile(
            existing_token, html=new_html
        )

        # Update only the title
        result = primitives.dashboards.update_tile(
            tile_token, title="Revenue by Category (Updated Q4)"
        )

        # Update title and description without touching HTML
        result = primitives.dashboards.update_tile(
            tile_token,
            title="Arrears Summary",
            description="Weekly arrears breakdown by region. Updated 2025-11-01.",
        )

        # Full update: new HTML, title, and description
        result = primitives.dashboards.update_tile(
            tile_token,
            html=regenerated_html,
            title="Sales Dashboard v2",
            description="Redesigned with stacked bar chart",
        )

        Anti-patterns
        -------------
        - WRONG: Creating a new tile when the existing one just needs a
          data refresh.
          CORRECT: Use ``update_tile()`` to replace the HTML -- the token
          and URL remain stable, so dashboard references keep working.

        - WRONG: Passing empty string for ``html`` to "clear" a tile.
          CORRECT: Always provide valid, renderable HTML.

        - WRONG: Updating a tile without checking ``result.succeeded``.
          CORRECT: Always verify the result before assuming the update took.

        Notes
        -----
        - The ``updated_at`` timestamp on the tile record is set to the
          current time on every successful update.
        - Updating ``html`` does NOT change ``data_bindings``.  To change
          data bindings, delete and recreate the tile.
        - Dashboards referencing this tile will reflect the update on their
          next render -- no dashboard update needed.

        See Also
        --------
        create_tile : Create a new tile from scratch.
        get_tile : Retrieve current tile content before updating.
        delete_tile : Remove a tile entirely.
        """

    @abstractmethod
    def delete_tile(self, token: str) -> bool:
        """
        Delete a tile and its token registration.

        **WARNING**: This is a destructive operation.  The tile's HTML content,
        metadata, and token registration are permanently removed.  The
        shareable URL will stop working.

        Dashboards that reference this tile are **not** automatically updated.
        Their layout JSON will still contain the deleted tile's token, but
        the tile will not render.  Update or recreate affected dashboards
        after deleting tiles.

        Parameters
        ----------
        token : str
            The 12-character URL-safe token of the tile to delete.

            Example: ``"Ab3xK9mP2qLz"``

        Returns
        -------
        bool
            ``True`` if the tile was found and deleted.  ``False`` if no
            tile with the given token exists (idempotent -- calling delete
            on an already-deleted token does not raise an error).

        Usage Examples
        --------------
        # Delete a specific tile
        deleted = primitives.dashboards.delete_tile("Ab3xK9mP2qLz")
        if deleted:
            print("Tile deleted")
        else:
            print("Tile not found (already deleted?)")

        # Clean up after replacing a tile with a new version
        new_result = primitives.dashboards.create_tile(
            new_html, title="Revenue Chart v2"
        )
        if new_result.succeeded:
            primitives.dashboards.delete_tile(old_token)

        # Delete tile and update dashboard to remove stale reference
        primitives.dashboards.delete_tile(stale_token)
        remaining_tiles = [t for t in layout if t.tile_token != stale_token]
        primitives.dashboards.update_dashboard(
            dashboard_token, tiles=remaining_tiles
        )

        Anti-patterns
        -------------
        - WRONG: Deleting a tile without updating dashboards that use it.
          CORRECT: After deleting, update or recreate any dashboards that
          referenced the deleted tile to avoid broken grid cells.

        - WRONG: Deleting and recreating a tile to update it.
          CORRECT: Use ``update_tile()`` to modify in place -- this
          preserves the token and URL, keeping dashboard references intact.

        Notes
        -----
        - Deletion is immediate and permanent.
        - The token is deregistered; the same token will not be reissued.
        - The tile's row is removed from the ``Dashboards/Tiles`` context.
        - Returns ``False`` (not an error) when the token is not found.

        See Also
        --------
        update_tile : Modify a tile without deleting it.
        delete_dashboard : Delete a dashboard (does not delete its tiles).
        list_tiles : Discover existing tiles before cleanup.
        """

    @abstractmethod
    def list_tiles(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[TileRecord]:
        """
        List tiles with metadata, optionally filtered.

        Use this to discover existing tiles, check what visualizations are
        available, or find tiles by title or description before composing
        them into a dashboard.  This is the primary discovery mechanism for
        tiles.

        Returns tile metadata **without** ``html_content`` to keep payloads
        small.  The ``html_content`` field is set to an empty string in
        every returned record.  Use ``get_tile()`` to retrieve the full
        HTML for a specific tile.

        Parameters
        ----------
        filter : str | None, default ``None``
            Python filter expression evaluated per tile row.  Uses the same
            syntax as DataManager's ``filter()`` method -- column names in
            scope, string values must be quoted.

            Filterable columns:
            - ``title`` (str): Tile title.
            - ``description`` (str | None): Tile description.
            - ``token`` (str): 12-char tile token.
            - ``has_data_bindings`` (bool): Whether the tile uses live data.
            - ``data_binding_contexts`` (str | None): Comma-separated contexts.
            - ``created_at`` (str): ISO-8601 creation timestamp.
            - ``updated_at`` (str): ISO-8601 last-update timestamp.

            Examples:
            - ``"title == 'Revenue by Category'"`` -- exact match
            - ``"has_data_bindings == True"`` -- only live-data tiles
            - ``"created_at >= '2025-01-01'"`` -- tiles created this year

            When ``None``, all tiles are returned (subject to ``limit``).

        limit : int, default ``50``
            Maximum number of tiles to return.  Use a higher value if you
            expect many tiles, but be mindful of payload size.

        Returns
        -------
        list[TileRecord]
            List of tile records with metadata.  Each ``TileRecord``
            contains:

            - ``tile_id`` (int | None): Auto-incremented numeric identifier.
            - ``token`` (str): The 12-char URL-safe token.
            - ``title`` (str): Human-readable tile title.
            - ``description`` (str | None): Optional longer description.
            - ``html_content`` (str): **Always empty string** in list results.
              Use ``get_tile(token)`` for full HTML.
            - ``has_data_bindings`` (bool): Whether the tile uses live data.
            - ``data_binding_contexts`` (str | None): Comma-separated context
              paths.
            - ``created_at`` (str | None): ISO-8601 creation timestamp.
            - ``updated_at`` (str | None): ISO-8601 last-update timestamp.

            Returns an empty list if no tiles match.

        Usage Examples
        --------------
        # List all tiles
        tiles = primitives.dashboards.list_tiles()
        for tile in tiles:
            print(f"[{tile.token}] {tile.title}")

        # Find tiles by title
        revenue_tiles = primitives.dashboards.list_tiles(
            filter="'revenue' in title.lower()"
        )

        # Find live-data tiles only
        live_tiles = primitives.dashboards.list_tiles(
            filter="has_data_bindings == True"
        )

        # Find tiles created after a specific date
        recent = primitives.dashboards.list_tiles(
            filter="created_at >= '2025-10-01'",
            limit=100,
        )

        # Check if a specific tile exists by token
        matches = primitives.dashboards.list_tiles(
            filter="token == 'Ab3xK9mP2qLz'"
        )
        exists = len(matches) > 0

        Anti-patterns
        -------------
        - WRONG: Using ``list_tiles`` then reading ``html_content`` from
          the results.
          CORRECT: ``html_content`` is always empty in list results.  Use
          ``get_tile(token)`` to fetch the full HTML for a specific tile.

        - WRONG: Calling ``get_tile`` in a loop for every tile.
          CORRECT: Use ``list_tiles()`` for batch metadata, then
          ``get_tile()`` only for the specific tiles whose HTML you need.

        - WRONG: Using ``list_tiles`` without a filter when you know what
          you're looking for.
          CORRECT: Provide a filter to narrow results and reduce payload.

        Notes
        -----
        - The ``html_content`` exclusion is intentional: tile HTML can be
          very large (megabytes for baked-in data), and listing many tiles
          with full HTML would be prohibitively expensive.
        - Results are not guaranteed in any particular order.
        - The ``limit`` applies after filtering.

        See Also
        --------
        get_tile : Retrieve full tile content by token.
        create_tile : Create a new tile.
        list_dashboards : List dashboards (analogous method).
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Dashboards
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def create_dashboard(
        self,
        title: str,
        *,
        description: Optional[str] = None,
        tiles: Optional[List[TilePosition]] = None,
    ) -> DashboardResult:
        """
        Create a new dashboard layout composed of existing tiles.

        A dashboard is a named composition of tiles arranged on a responsive
        12-column grid.  Each tile is placed by specifying its token and
        grid coordinates (``x``, ``y``, ``w``, ``h``).  The dashboard is
        stored in the ``Dashboards/Layouts`` Unify context and assigned a
        12-character URL-safe token for shareable access.

        Tiles must be created **before** the dashboard -- use
        ``create_tile()`` first, then reference the returned tokens in the
        ``TilePosition`` list.

        Parameters
        ----------
        title : str
            Human-readable dashboard title.  Displayed in the dashboard
            header and used for discovery via ``list_dashboards()``.

            Examples: ``"Sales Overview"``, ``"Q4 KPI Dashboard"``,
            ``"Arrears Monitoring"``

        description : str | None, default ``None``
            Longer description of the dashboard's purpose, audience, or data
            sources.  Useful for discoverability when browsing dashboards.
            When ``None``, only the title is stored.

            Example: ``"Executive summary of Q4 sales performance across all
            regions.  Contains revenue KPIs, trend charts, and a detail
            table."``

        tiles : list[TilePosition] | None, default ``None``
            Layout positions for each tile on the 12-column grid.  Each
            ``TilePosition`` specifies:

            - ``tile_token`` (str, required): The 12-char token of an
              existing tile (from ``TileResult.token``).
            - ``x`` (int, default ``0``): Column offset (0--11).  The
              leftmost column is 0.
            - ``y`` (int, default ``0``): Row offset in grid row units.
              Rows stack vertically; use higher ``y`` values for tiles
              below others.
            - ``w`` (int, default ``6``): Width in columns (1--12).  A
              value of 12 spans the full width; 6 is half-width.
            - ``h`` (int, default ``4``): Height in row units.

            Layout examples::

                # Full-width header KPI row + two half-width charts below
                tiles=[
                    TilePosition(tile_token=kpi.token, x=0, y=0, w=12, h=2),
                    TilePosition(tile_token=chart1.token, x=0, y=2, w=6, h=4),
                    TilePosition(tile_token=chart2.token, x=6, y=2, w=6, h=4),
                ]

                # Three-column layout
                tiles=[
                    TilePosition(tile_token=a.token, x=0, y=0, w=4, h=4),
                    TilePosition(tile_token=b.token, x=4, y=0, w=4, h=4),
                    TilePosition(tile_token=c.token, x=8, y=0, w=4, h=4),
                ]

            When ``None``, an empty dashboard is created (no tiles).  Tiles
            can be added later via ``update_dashboard()``.

        Returns
        -------
        DashboardResult
            Result object with the following fields:

            - ``url`` (str | None): Shareable URL for viewing the dashboard.
              ``None`` on failure.
            - ``token`` (str | None): The 12-character URL-safe token
              assigned to this dashboard.  Use this for ``get_dashboard()``,
              ``update_dashboard()``, and ``delete_dashboard()``.
            - ``title`` (str | None): The title that was stored.
            - ``tiles`` (list[TilePosition] | None): The tile positions
              that were stored (echoed back for confirmation).
            - ``error`` (str | None): Error message if the operation failed.
              ``None`` on success.
            - ``succeeded`` (bool, property): ``True`` if ``url`` is not
              ``None`` and ``error`` is ``None``.

        Raises
        ------
        ValueError
            If ``title`` is empty.
        ValueError
            If any ``tile_token`` in ``tiles`` does not reference an
            existing tile.

        Usage Examples
        --------------
        # Create a dashboard with a KPI row and a chart row
        kpi = primitives.dashboards.create_tile(kpi_html, title="KPI Card")
        chart = primitives.dashboards.create_tile(chart_html, title="Revenue Chart")
        table = primitives.dashboards.create_tile(table_html, title="Detail Table")

        result = primitives.dashboards.create_dashboard(
            "Sales Overview",
            description="Q4 sales summary with KPIs, trend chart, and detail table",
            tiles=[
                TilePosition(tile_token=kpi.token, x=0, y=0, w=4, h=2),
                TilePosition(tile_token=chart.token, x=4, y=0, w=8, h=4),
                TilePosition(tile_token=table.token, x=0, y=4, w=12, h=4),
            ],
        )
        print(result.url)  # shareable URL for the dashboard

        # Create an empty dashboard (tiles added later)
        result = primitives.dashboards.create_dashboard(
            "WIP Dashboard",
            description="Work in progress -- tiles will be added incrementally",
        )

        # Full-width single tile dashboard
        result = primitives.dashboards.create_dashboard(
            "Arrears Heatmap",
            tiles=[
                TilePosition(tile_token=heatmap.token, x=0, y=0, w=12, h=8),
            ],
        )

        # Three-column equal-width layout
        result = primitives.dashboards.create_dashboard(
            "Regional Comparison",
            tiles=[
                TilePosition(tile_token=east.token, x=0, y=0, w=4, h=4),
                TilePosition(tile_token=west.token, x=4, y=0, w=4, h=4),
                TilePosition(tile_token=north.token, x=8, y=0, w=4, h=4),
            ],
        )

        Anti-patterns
        -------------
        - WRONG: Referencing tile tokens that haven't been created yet.
          CORRECT: Always create tiles with ``create_tile()`` first, then
          use the returned tokens in ``TilePosition``.

        - WRONG: Setting ``w`` values that exceed 12 or ``x + w > 12``.
          CORRECT: Ensure tiles fit within the 12-column grid.  A tile at
          ``x=8`` can have ``w`` at most ``4`` (8 + 4 = 12).

        - WRONG: Creating a new dashboard to update the layout.
          CORRECT: Use ``update_dashboard()`` to modify an existing
          dashboard's layout in place.

        - WRONG: Not checking ``result.succeeded`` before using the URL.
          CORRECT: Always verify ``result.succeeded`` is ``True``.

        Notes
        -----
        - The dashboard token is generated server-side and is unique.
        - The layout is stored as a JSON-serialized list of ``TilePosition``
          objects in the ``layout`` column of the dashboard record.
        - ``tile_count`` is automatically computed from the tiles list.
        - Dashboards and tiles have independent lifecycles: deleting a
          dashboard does **not** delete its tiles.
        - The same tile can appear in multiple dashboards and at multiple
          positions within a single dashboard.

        See Also
        --------
        create_tile : Create tiles before composing them into a dashboard.
        update_dashboard : Modify a dashboard's layout or metadata.
        get_dashboard : Retrieve a dashboard by token.
        list_dashboards : Discover existing dashboards.
        """

    @abstractmethod
    def get_dashboard(self, token: str) -> Optional[DashboardResult]:
        """
        Retrieve a dashboard by its token.

        Use this to fetch a dashboard's metadata and tile layout, for example
        to inspect the current layout before updating it, to extract tile
        tokens for reuse, or to verify what was stored.

        Parameters
        ----------
        token : str
            The 12-character URL-safe token identifying the dashboard.  This
            is the value returned in ``DashboardResult.token`` from
            ``create_dashboard()`` or ``update_dashboard()``.

            Example: ``"Xy7mN2pQ9rLz"``

        Returns
        -------
        DashboardResult | None
            The dashboard result when found, or ``None`` if no dashboard
            exists with the given token.

            The ``DashboardResult`` contains:

            - ``url`` (str | None): Shareable URL for viewing the dashboard.
            - ``token`` (str | None): The 12-char URL-safe token.
            - ``title`` (str | None): Human-readable dashboard title.
            - ``tiles`` (list[TilePosition] | None): The tile layout
              positions.  Each ``TilePosition`` has ``tile_token``, ``x``,
              ``y``, ``w``, ``h``.
            - ``error`` (str | None): Error message (should be ``None``
              for a successful retrieval).
            - ``succeeded`` (bool, property): ``True`` when retrieval
              succeeded.

        Usage Examples
        --------------
        # Retrieve and inspect a dashboard layout
        dashboard = primitives.dashboards.get_dashboard("Xy7mN2pQ9rLz")
        if dashboard is not None:
            print(f"Title: {dashboard.title}")
            print(f"URL: {dashboard.url}")
            for tile_pos in (dashboard.tiles or []):
                print(f"  Tile {tile_pos.tile_token} at ({tile_pos.x}, {tile_pos.y})"
                      f" size {tile_pos.w}x{tile_pos.h}")

        # Extract tile tokens from an existing dashboard
        dashboard = primitives.dashboards.get_dashboard(token)
        if dashboard is not None and dashboard.tiles:
            tile_tokens = [t.tile_token for t in dashboard.tiles]

        # Check existence before updating
        if primitives.dashboards.get_dashboard(token) is None:
            print("Dashboard not found -- creating a new one")

        Anti-patterns
        -------------
        - WRONG: Calling ``get_dashboard`` in a loop to list all dashboards.
          CORRECT: Use ``list_dashboards()`` for batch discovery.

        - WRONG: Using ``get_dashboard`` just to check existence.
          CORRECT: Use ``list_dashboards(filter="token == '...'")`` for
          lightweight existence checks when you don't need the full result.

        Notes
        -----
        - Returns ``None`` (not an error) when the token is not found.
          Always check the return value before accessing fields.
        - The ``tiles`` field contains the deserialized layout from the
          ``layout`` JSON column in the dashboard record.

        See Also
        --------
        list_dashboards : List dashboards without fetching each one.
        update_dashboard : Modify a dashboard's layout or metadata.
        create_dashboard : Create a new dashboard.
        """

    @abstractmethod
    def update_dashboard(
        self,
        token: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tiles: Optional[List[TilePosition]] = None,
    ) -> DashboardResult:
        """
        Update an existing dashboard's metadata or layout.

        Use this to rearrange tiles, add or remove tiles from the grid,
        or change the dashboard's title or description -- without changing
        the dashboard's token or URL.  Only the fields you provide are
        updated; omitted fields retain their current values.

        **Important**: When ``tiles`` is provided, it **replaces the entire
        layout**.  To add a tile to an existing layout, first retrieve the
        current layout with ``get_dashboard()``, append the new position,
        then pass the full list to ``update_dashboard()``.

        Parameters
        ----------
        token : str
            The 12-character URL-safe token of the dashboard to update.
            Must reference an existing dashboard.

            Example: ``"Xy7mN2pQ9rLz"``

        title : str | None, default ``None``
            New title for the dashboard.  When ``None``, the existing title
            is preserved.

        description : str | None, default ``None``
            New description for the dashboard.  When ``None``, the existing
            description is preserved.

        tiles : list[TilePosition] | None, default ``None``
            New tile layout positions.  **Replaces the entire layout** --
            this is not a merge or append operation.

            Each ``TilePosition`` specifies:
            - ``tile_token`` (str): Token of an existing tile.
            - ``x`` (int): Column offset (0--11).
            - ``y`` (int): Row offset.
            - ``w`` (int): Width in columns (1--12).
            - ``h`` (int): Height in row units.

            When ``None``, the existing layout is preserved.

            Pass an empty list ``[]`` to clear all tiles from the dashboard.

        Returns
        -------
        DashboardResult
            Result object with the same structure as ``create_dashboard()``:

            - ``url`` (str | None): The (unchanged) shareable URL.
            - ``token`` (str | None): The (unchanged) 12-char token.
            - ``title`` (str | None): The current title (updated if provided).
            - ``tiles`` (list[TilePosition] | None): The current layout
              (updated if provided).
            - ``error`` (str | None): Error message if the update failed.
            - ``succeeded`` (bool, property): ``True`` on success.

        Raises
        ------
        ValueError
            If the token does not reference an existing dashboard.
        ValueError
            If any ``tile_token`` in ``tiles`` does not reference an
            existing tile.

        Usage Examples
        --------------
        # Add a tile to an existing dashboard layout
        dashboard = primitives.dashboards.get_dashboard(dash_token)
        current_tiles = list(dashboard.tiles or [])
        current_tiles.append(
            TilePosition(tile_token=new_tile.token, x=0, y=8, w=12, h=4)
        )
        result = primitives.dashboards.update_dashboard(
            dash_token, tiles=current_tiles
        )

        # Remove a tile from the layout
        dashboard = primitives.dashboards.get_dashboard(dash_token)
        filtered = [t for t in (dashboard.tiles or [])
                    if t.tile_token != tile_to_remove]
        result = primitives.dashboards.update_dashboard(
            dash_token, tiles=filtered
        )

        # Rearrange tiles (make chart full-width)
        result = primitives.dashboards.update_dashboard(
            dash_token,
            tiles=[
                TilePosition(tile_token=kpi.token, x=0, y=0, w=12, h=2),
                TilePosition(tile_token=chart.token, x=0, y=2, w=12, h=6),
            ],
        )

        # Update only the title (layout unchanged)
        result = primitives.dashboards.update_dashboard(
            dash_token, title="Sales Dashboard (Final)"
        )

        # Clear all tiles from the dashboard
        result = primitives.dashboards.update_dashboard(
            dash_token, tiles=[]
        )

        Anti-patterns
        -------------
        - WRONG: Passing only the new tile when adding to a layout.
          CORRECT: ``tiles`` replaces the entire layout.  Retrieve the
          current layout, append the new tile, and pass the full list.

        - WRONG: Creating a new dashboard to update the layout.
          CORRECT: Use ``update_dashboard()`` to modify in place -- this
          preserves the token and URL.

        - WRONG: Not checking ``result.succeeded`` after updating.
          CORRECT: Always verify the result before assuming the update took.

        Notes
        -----
        - The ``updated_at`` timestamp on the dashboard record is set to
          the current time on every successful update.
        - The ``tile_count`` is automatically recomputed when ``tiles`` is
          provided.
        - The dashboard's token and URL remain stable across updates.
        - Providing ``tiles=[]`` creates an empty dashboard (zero tiles).

        See Also
        --------
        get_dashboard : Retrieve the current layout before updating.
        create_dashboard : Create a new dashboard from scratch.
        delete_dashboard : Remove a dashboard entirely.
        """

    @abstractmethod
    def delete_dashboard(self, token: str) -> bool:
        """
        Delete a dashboard and its token registration.

        **Does NOT delete the individual tiles** -- they remain accessible
        via their own tokens and can be reused in other dashboards.  Only
        the dashboard record and its layout JSON are removed.

        Parameters
        ----------
        token : str
            The 12-character URL-safe token of the dashboard to delete.

            Example: ``"Xy7mN2pQ9rLz"``

        Returns
        -------
        bool
            ``True`` if the dashboard was found and deleted.  ``False`` if
            no dashboard with the given token exists (idempotent -- calling
            delete on an already-deleted token does not raise an error).

        Usage Examples
        --------------
        # Delete a specific dashboard
        deleted = primitives.dashboards.delete_dashboard("Xy7mN2pQ9rLz")
        if deleted:
            print("Dashboard deleted (tiles still exist)")
        else:
            print("Dashboard not found")

        # Replace a dashboard with a new version
        primitives.dashboards.delete_dashboard(old_dash_token)
        new_result = primitives.dashboards.create_dashboard(
            "Sales Overview v2",
            tiles=updated_layout,
        )

        # Delete dashboard and optionally clean up orphaned tiles
        dashboard = primitives.dashboards.get_dashboard(token)
        tile_tokens = [t.tile_token for t in (dashboard.tiles or [])]
        primitives.dashboards.delete_dashboard(token)
        for tile_token in tile_tokens:
            primitives.dashboards.delete_tile(tile_token)

        Anti-patterns
        -------------
        - WRONG: Assuming tile deletion happens automatically with dashboard
          deletion.
          CORRECT: Tiles have independent lifecycles.  Delete tiles
          separately if they are no longer needed.

        - WRONG: Deleting and recreating a dashboard to update it.
          CORRECT: Use ``update_dashboard()`` to modify in place -- this
          preserves the token and URL.

        Notes
        -----
        - Deletion is immediate and permanent.
        - The dashboard token is deregistered; the shareable URL stops
          working.
        - The dashboard row is removed from the ``Dashboards/Layouts``
          context.
        - Returns ``False`` (not an error) when the token is not found.
        - Tiles referenced by the deleted dashboard are unaffected.

        See Also
        --------
        update_dashboard : Modify a dashboard without deleting it.
        delete_tile : Delete individual tiles.
        list_dashboards : Discover existing dashboards before cleanup.
        """

    @abstractmethod
    def list_dashboards(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[DashboardRecord]:
        """
        List dashboards with metadata, optionally filtered.

        Use this to discover existing dashboards, find a specific dashboard
        by title, or enumerate all dashboards before cleanup.  This is the
        primary discovery mechanism for dashboards.

        Parameters
        ----------
        filter : str | None, default ``None``
            Python filter expression evaluated per dashboard row.  Uses the
            same syntax as DataManager's ``filter()`` method -- column names
            in scope, string values must be quoted.

            Filterable columns:
            - ``title`` (str): Dashboard title.
            - ``description`` (str | None): Dashboard description.
            - ``token`` (str): 12-char dashboard token.
            - ``tile_count`` (int): Number of tiles in the layout.
            - ``created_at`` (str): ISO-8601 creation timestamp.
            - ``updated_at`` (str): ISO-8601 last-update timestamp.

            Examples:
            - ``"title == 'Sales Overview'"`` -- exact match
            - ``"tile_count > 3"`` -- dashboards with many tiles
            - ``"created_at >= '2025-01-01'"`` -- recent dashboards

            When ``None``, all dashboards are returned (subject to ``limit``).

        limit : int, default ``50``
            Maximum number of dashboards to return.  Use a higher value if
            you expect many dashboards.

        Returns
        -------
        list[DashboardRecord]
            List of dashboard records.  Each ``DashboardRecord`` contains:

            - ``dashboard_id`` (int | None): Auto-incremented numeric
              identifier.
            - ``token`` (str): The 12-char URL-safe token.
            - ``title`` (str): Human-readable dashboard title.
            - ``description`` (str | None): Optional longer description.
            - ``layout`` (str): JSON-serialized list of ``TilePosition``
              objects.  Parse with ``json.loads()`` if you need the
              positions; or use ``get_dashboard()`` which returns
              deserialized ``TilePosition`` objects directly.
            - ``tile_count`` (int): Number of tiles in the dashboard.
            - ``created_at`` (str | None): ISO-8601 creation timestamp.
            - ``updated_at`` (str | None): ISO-8601 last-update timestamp.

            Returns an empty list if no dashboards match.

        Usage Examples
        --------------
        # List all dashboards
        dashboards = primitives.dashboards.list_dashboards()
        for db in dashboards:
            print(f"[{db.token}] {db.title} ({db.tile_count} tiles)")

        # Find dashboards by title
        sales = primitives.dashboards.list_dashboards(
            filter="'sales' in title.lower()"
        )

        # Find dashboards with more than 3 tiles
        complex_dbs = primitives.dashboards.list_dashboards(
            filter="tile_count > 3"
        )

        # Find a specific dashboard by token
        matches = primitives.dashboards.list_dashboards(
            filter="token == 'Xy7mN2pQ9rLz'"
        )
        if matches:
            print(f"Found: {matches[0].title}")

        # Find recent dashboards
        recent = primitives.dashboards.list_dashboards(
            filter="created_at >= '2025-10-01'",
            limit=100,
        )

        Anti-patterns
        -------------
        - WRONG: Calling ``get_dashboard`` in a loop for every dashboard.
          CORRECT: Use ``list_dashboards()`` for batch metadata, then
          ``get_dashboard()`` only for dashboards whose full layout you need
          as deserialized ``TilePosition`` objects.

        - WRONG: Parsing the ``layout`` JSON from ``list_dashboards``
          results when you need typed ``TilePosition`` objects.
          CORRECT: Use ``get_dashboard(token)`` which returns deserialized
          ``DashboardResult.tiles`` with proper typing.

        Notes
        -----
        - The ``layout`` field is a raw JSON string in ``DashboardRecord``.
          Use ``get_dashboard()`` for typed ``TilePosition`` objects.
        - Results are not guaranteed in any particular order.
        - The ``limit`` applies after filtering.

        See Also
        --------
        get_dashboard : Retrieve full dashboard with deserialized layout.
        create_dashboard : Create a new dashboard.
        list_tiles : List tiles (analogous method).
        """
