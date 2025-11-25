"""Setup script to download Northwind database and create views."""
import urllib.request
import sqlite3
from pathlib import Path


def download_database(force_redownload=False):
    """Download Northwind database."""
    db_path = Path("data/northwind.sqlite")
    
    if db_path.exists() and not force_redownload:
        # Verify database is valid
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
            cursor.fetchone()
            conn.close()
            print(f"Database already exists and is valid at {db_path}")
            return str(db_path)
        except Exception as e:
            print(f"Existing database is corrupted: {e}")
            print("Re-downloading...")
            db_path.unlink()  # Delete corrupted file
    
    print("Downloading Northwind database...")
    url = "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db"
    
    try:
        print("This may take a few minutes...")
        urllib.request.urlretrieve(url, db_path)
        
        # Verify download
        if db_path.exists() and db_path.stat().st_size > 1000000:  # At least 1MB
            print(f"✓ Database downloaded to {db_path} ({db_path.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            raise Exception("Downloaded file is too small or missing")
            
    except Exception as e:
        print(f"Error downloading database: {e}")
        print("Please download manually from:")
        print(url)
        if db_path.exists():
            db_path.unlink()  # Remove partial download
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
    import sys
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Check if force re-download is requested
    force = "--force" in sys.argv or "-f" in sys.argv
    
    # Download database
    db_path = download_database(force_redownload=force)
    
    if db_path:
        # Create views
        create_views(db_path)
        print("\n✓ Setup complete!")
    else:
        print("\n✗ Setup failed")
        sys.exit(1)

