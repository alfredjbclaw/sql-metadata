from typing import Any

from sqlglot import exp
from sqlglot.dialects.dialect import DialectType

from sql_metadata.column_extractor import _is_date_part_unit
from sql_metadata.table_extractor import TableExtractor
from sql_metadata.utils import UniqueList


_SELECT_CLAUSES: dict[str, str] = {
    "where": "where",
    "group": "group_by",
    "order": "order_by",
    "having": "having",
}


class TableColumnExtractor:
    def __init__(
        self,
        ast: exp.Expression,
        table_aliases: dict[str, str],
        dialect: DialectType = None,
    ) -> None:
        self._ast = ast
        self._table_aliases = table_aliases
        self._table_extractor = TableExtractor(ast, dialect=dialect)
        self._columns: dict[str, dict[str, UniqueList]] = {}

    def extract(self) -> dict[str, dict[str, UniqueList]]:
        self._walk_query(self._ast)
        return self._columns

    def _walk_query(self, node: exp.Expression) -> None:
        if isinstance(node, exp.Select):
            self._walk_select(node)
            return

        for child in node.iter_expressions():
            self._walk_query(child)

    def _walk_select(self, select: exp.Select) -> None:
        tables, aliases = self._select_tables(select)

        for expression in select.expressions or []:
            self._walk_clause(expression, "select", tables, aliases)

        for key, clause in _SELECT_CLAUSES.items():
            child = select.args.get(key)
            if child is not None:
                self._walk_clause(child, clause, tables, aliases)

        for join in select.args.get("joins") or []:
            self._walk_join(join, tables, aliases)

        self._walk_source_queries(select.args.get("from") or select.args.get("from_"))
        for join in select.args.get("joins") or []:
            self._walk_source_queries(join.this)

    def _walk_join(
        self, join: exp.Join, tables: list[str], aliases: dict[str, str]
    ) -> None:
        on_clause = join.args.get("on")
        if on_clause is not None:
            self._walk_clause(on_clause, "join", tables, aliases)

        using_columns = join.args.get("using") or []
        for column in using_columns:
            if hasattr(column, "name"):
                for table in tables:
                    self._add_column("join", table, column.name)

    def _walk_clause(
        self,
        node: Any,
        clause: str,
        tables: list[str],
        aliases: dict[str, str],
    ) -> None:
        if isinstance(node, list):
            for item in node:
                self._walk_clause(item, clause, tables, aliases)
            return

        if not isinstance(node, exp.Expression):
            return

        if isinstance(node, exp.Select):
            self._walk_select(node)
            return

        if isinstance(node, exp.Column):
            if not _is_date_part_unit(node):
                table = self._table_for_column(node, tables, aliases)
                if table is not None:
                    self._add_column(clause, table, node.name.rstrip("#"))
            return

        for child in node.iter_expressions():
            self._walk_clause(child, clause, tables, aliases)

    def _select_tables(self, select: exp.Select) -> tuple[list[str], dict[str, str]]:
        tables = UniqueList()
        aliases: dict[str, str] = {}
        from_clause = select.args.get("from") or select.args.get("from_")

        for table in self._direct_tables(from_clause):
            name = self._table_extractor._table_full_name(table)
            if not name:
                continue
            tables.append(name)
            if table.alias:
                aliases[table.alias] = name

        for join in select.args.get("joins") or []:
            for table in self._direct_tables(join.this):
                name = self._table_extractor._table_full_name(table)
                if not name:
                    continue
                tables.append(name)
                if table.alias:
                    aliases[table.alias] = name

        return list(tables), aliases

    def _direct_tables(self, source: Any) -> list[exp.Table]:
        if source is None:
            return []

        if isinstance(source, exp.Table):
            return [source]

        if isinstance(source, exp.From):
            tables = []
            if source.this is not None:
                tables.extend(self._direct_tables(source.this))
            for expression in source.expressions or []:
                tables.extend(self._direct_tables(expression))
            return tables

        if isinstance(source, exp.Alias):
            return self._direct_tables(source.this)

        return []

    def _walk_source_queries(self, source: Any) -> None:
        if source is None or isinstance(source, exp.Table):
            return

        if isinstance(source, exp.From):
            self._walk_source_queries(source.this)
            for expression in source.expressions or []:
                self._walk_source_queries(expression)
            return

        if isinstance(source, exp.Subquery):
            self._walk_query(source.this)
            return

        if isinstance(source, exp.Expression):
            for child in source.iter_expressions():
                self._walk_source_queries(child)

    def _table_for_column(
        self,
        column: exp.Column,
        tables: list[str],
        aliases: dict[str, str],
    ) -> str | None:
        if column.table:
            return self._qualified_column_table(column, aliases)
        if len(tables) == 1:
            return tables[0]
        return None

    def _qualified_column_table(
        self, column: exp.Column, aliases: dict[str, str]
    ) -> str:
        if column.table in aliases:
            return aliases[column.table]
        if column.table in self._table_aliases:
            return self._table_aliases[column.table]

        parts = []
        for key in ("catalog", "db"):
            value = column.args.get(key)
            if value is not None:
                parts.append(self._name(value))
        parts.append(column.table)
        return ".".join(parts)

    @staticmethod
    def _name(node: Any) -> str:
        if isinstance(node, exp.Expression):
            return node.name
        return str(node)

    def _add_column(self, clause: str, table: str, column: str) -> None:
        self._columns.setdefault(clause, {}).setdefault(table, UniqueList()).append(
            column
        )
