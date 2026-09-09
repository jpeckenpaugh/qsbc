import os

from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    conninfo=(
        f"host={os.environ.get('PGHOST', 'localhost')} "
        f"port={os.environ.get('PGPORT', '5432')} "
        f"dbname={os.environ.get('PGDATABASE', 'c3pa')} "
        f"user={os.environ.get('PGUSER', 'c3pa')} "
        f"password={os.environ.get('PGPASSWORD', 'c3pa')}"
    ),
    min_size=1,
    max_size=8,
    open=False,
)


def connect():
    pool.open()
    return pool