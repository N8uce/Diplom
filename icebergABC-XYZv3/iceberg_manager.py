### Updated `iceberg_manager.py`:
import os
import time
import shutil
import logging
from datetime import datetime
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema, NestedField
from pyiceberg.types import IntegerType, StringType, FloatType
from pyiceberg.types import FixedType
from pyiceberg.types import ListType
from pyiceberg.exceptions import NoSuchTableError, NamespaceAlreadyExistsError
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl
logger = logging.getLogger(__name__)

class IcebergManager:
    NAMESPACE = "default"
    def __init__(
        self,
        test_id: str = None,
        catalog_type: str = "rest-minio",
        warehouse_base: str = None,
        rest_uri: str = None
    ):
        self.catalog_type = catalog_type
        self.test_id = test_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = warehouse_base or os.getcwd()
        self.warehouse_path = os.path.abspath(
            os.path.join(base, f"iceberg_warehouse_{self.test_id}")
        )
        os.makedirs(self.warehouse_path, exist_ok=True)
        self.table_name = f"sales_{self.test_id}"

        #Настройки для разных типов
        if catalog_type == "rest-minio":
            # REST‑каталог + MinIO S3
            uri = rest_uri or "http://localhost:8181"
            catalog_props = {
                "type": "rest",
                "uri": uri,
                "warehouse": "s3a://warehouse",
                "s3.endpoint": "http://localhost:9000",
                "s3.access-key-id": "admin",
                "s3.secret-access-key": "password",
                "s3.path-style-access": "true",
                "io-impl": "org.apache.iceberg.aws.s3.S3FileIO"
            }
        elif catalog_type == "sql":

            catalog_props = {
                "type": "sql",
                "uri": "postgresql+psycopg2://postgres:1234@localhost/diplom",
                "warehouse": f"file://E:/iceberg/warehouse"

            }
        else:
            raise ValueError(f"Unknown catalog_type: {catalog_type}")

        # Загружаем каталог
        self.catalog = load_catalog("default", **catalog_props)
        self._ensure_table()


    def _ensure_table(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                try:
                    if self.NAMESPACE not in self.catalog.list_namespaces():
                        self.catalog.create_namespace(self.NAMESPACE)
                        logger.info(f"Created namespace: {self.NAMESPACE}")
                except NamespaceAlreadyExistsError:
                    logger.info(f"Namespace already exists: {self.NAMESPACE}")
                except Exception as e:
                    logger.warning(f"Could not ensure namespace: {e}")

                try:
                    self.catalog.drop_table(f"{self.NAMESPACE}.{self.table_name}")
                    logger.info(f"Dropped existing table {self.table_name}")
                except NoSuchTableError:
                    pass

                schema = Schema(
                    NestedField(1, "id", IntegerType(), required=False),
                    NestedField(2, "product", StringType(), required=False),
                    NestedField(3, "customer", StringType(), required=False),
                    NestedField(4, "date", StringType(), required=False),
                    NestedField(5, "store", StringType(), required=False),
                    NestedField(6, "quantity", IntegerType(), required=False),
                    NestedField(7, "price", FloatType(), required=False),
                    NestedField(8, "revenue", FloatType(), required=False)
                )

                self.table = self.catalog.create_table(
                    identifier=f"{self.NAMESPACE}.{self.table_name}",
                    schema=schema,
                    properties={
                        "format-version": "2",
                        "write.parquet.compression-codec": "zstd"
                    }
                )
                logger.info(f"Created new table {self.table_name}")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to create table after {max_retries} attempts: {e}")
                    raise
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait}s...")
                time.sleep(wait)

    def cleanup(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.catalog.drop_table(f"{self.NAMESPACE}.{self.table_name}")
                shutil.rmtree(self.warehouse_path, ignore_errors=True)
                logger.info(f"Cleaned up {self.table_name}")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Final cleanup failed: {e}")
                else:
                    logger.warning(f"Cleanup attempt {attempt+1} failed, retrying...")
                    time.sleep(1)

    def save_sales(self, df):
        import pyarrow as pa

        df = df.with_columns(
            (df["quantity"] * df["price"]).alias("revenue")
        )

        table = df.to_arrow()

        arrow_schema = pa.schema([
            pa.field("id", pa.int32()),
            pa.field("product", pa.string()),
            pa.field("customer", pa.string()),
            pa.field("date", pa.string()),
            pa.field("store", pa.string()),
            pa.field("quantity", pa.int32()),
            pa.field("price", pa.float32()),
            pa.field("revenue", pa.float32()),
        ])

        table = table.cast(arrow_schema)
        self.table.append(table)
        logger.info(f"Saved {len(df)} rows to {self.table_name}")

    def read_sales(self) -> pl.DataFrame:
        scan = self.table.scan()
        arrow_table = scan.to_arrow()
        df = pl.from_arrow(arrow_table)
        return df

