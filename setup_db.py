"""Setup script to download Northwind database and create views."""
import urllib.request
import sqlite3
from pathlib import Path


def download_database():
    """Download Northwind database."""
    db_path = Path("data/northwind.sqlite")
    
    if db_path.exists():
        print(f"Database already exists at {db_path}")
        return str(db_path)
    
    print("Downloading Northwind database...")
    url = "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db"
    
    try:
        urllib.request.urlretrieve(url, db_path)
        print(f"✓ Database downloaded to {db_path}")
    except Exception as e:
        print(f"Error downloading database: {e}")
        print("Please download manually from:")
        print(url)
        return None
    
    return str(db_path)


def create_views(db_path: str):
    """Create lowercase compatibility views."""
    print("Creating lowercase compatibility views...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
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
            print(f"✓ Created view: {view_name}")
        except Exception as e:
            print(f"Error creating view {view_name}: {e}")
    
    conn.commit()
    conn.close()
    print("✓ Views created")


if __name__ == "__main__":
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Download database
    db_path = download_database()
    
    if db_path:
        # Create views
        create_views(db_path)
        print("\n✓ Setup complete!")
    else:
        print("\n✗ Setup failed")

