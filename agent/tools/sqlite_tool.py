"""SQLite tool for database access and schema introspection."""
import sqlite3
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


class SQLiteTool:
    """Tool for executing SQL queries and introspecting schema."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        self._ensure_lowercase_views()
    
    def _ensure_lowercase_views(self):
        """Create lowercase compatibility views if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Test if database is valid
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
            cursor.fetchone()
            
            views = [
                ("orders", "Orders"),
                ("order_items", '"Order Details"'),
                ("products", "Products"),
                ("customers", "Customers"),
            ]
            
            for view_name, table_name in views:
                try:
                    cursor.execute(
                        f"CREATE VIEW IF NOT EXISTS {view_name} AS SELECT * FROM {table_name};"
                    )
                except sqlite3.OperationalError as e:
                    # View might already exist or table doesn't exist, skip
                    pass
            
            conn.commit()
            conn.close()
        except sqlite3.DatabaseError as e:
            raise FileNotFoundError(
                f"Database is corrupted or invalid: {e}. "
                f"Please re-download it by running: python setup_db.py"
            )
    
    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Get database schema information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        schema = {}
        
        # Get all tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Quote table name to handle spaces and special characters
            # SQLite requires double quotes for identifiers with spaces
            # Always quote to be safe
            quoted_table = f'"{table}"'
            cursor.execute(f"PRAGMA table_info({quoted_table})")
            columns = cursor.fetchall()
            schema[table] = [
                {
                    "name": col[1],
                    "type": col[2],
                    "notnull": bool(col[3]),
                    "default": col[4],
                    "pk": bool(col[5]),
                }
                for col in columns
            ]
        
        conn.close()
        return schema
    
    def get_schema_string(self) -> str:
        """Get schema as a formatted string for prompts."""
        schema = self.get_schema()
        lines = ["Database Schema:"]
        
        for table, columns in schema.items():
            lines.append(f"\n{table}:")
            for col in columns:
                pk_str = " (PRIMARY KEY)" if col["pk"] else ""
                notnull_str = " NOT NULL" if col["notnull"] else ""
                lines.append(f"  - {col['name']}: {col['type']}{pk_str}{notnull_str}")
        
        return "\n".join(lines)
    
    def execute(self, query: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], List[str]]:
        """
        Execute SQL query.
        
        Returns:
            (rows, error, columns): rows as list of dicts, error message if any, column names
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                result = [dict(row) for row in rows]
                conn.close()
                return result, None, columns
            else:
                conn.commit()
                conn.close()
                return None, None, []
        
        except sqlite3.Error as e:
            error_msg = str(e)
            conn.close()
            return None, error_msg, []
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables

