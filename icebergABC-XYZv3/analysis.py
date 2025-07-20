# analysis.py
import ray
import math
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl
from dataclasses import dataclass
from typing import List, Dict, Tuple
from data_generation import generate_sales_chunk


@ray.remote
def product_revenue_chunk(chunk_df: pl.DataFrame) -> Dict[str, float]:
    df = chunk_df.with_columns((pl.col("quantity") * pl.col("price")).alias("revenue"))
    grouped = df.group_by("product").agg(pl.col("revenue").sum().alias("sum_rev"))
    return {row["product"]: row["sum_rev"] for row in grouped.iter_rows(named=True)}

@ray.remote
def abc_analysis(chunks: List[ray.ObjectRef],
                                     thresholds: Tuple[float, float, float] = (0.8, 0.15, 0.05)
                                    ) -> Dict[str, List[str]]:
    # 1) Параллельно считаем словари {product: local_revenue}
    locals_list: List[Dict[str, float]] = ray.get([
        product_revenue_chunk.remote(c) for c in chunks
    ])

    # 2) Сливаем в глобальный словарь
    global_rev: Dict[str, float] = {}
    for loc in locals_list:
        for prod, rev in loc.items():
            global_rev[prod] = global_rev.get(prod, 0.0) + rev

    # 3) Сортируем продукты по выручке
    prod_items: List[Tuple[str, float]] = sorted(
        global_rev.items(), key=lambda x: x[1], reverse=True
    )

    # 4) Вычисляем кумулятивную долю
    total = sum(global_rev.values())
    cum = 0.0
    t1, t2 = thresholds[0], thresholds[0] + thresholds[1]
    cats = {"A": [], "B": [], "C": []}

    for prod, rev in prod_items:
        cum += rev
        frac = cum / total
        if frac <= t1:
            cats["A"].append(prod)
        elif frac <= t2:
            cats["B"].append(prod)
        else:
            cats["C"].append(prod)

    return cats

@ray.remote
def xyz_analysis_chunk(chunk_df: pl.DataFrame) -> Dict[str, Dict[str, float]]:

    #колонка month (из первых 7 символов даты 'YYYY-MM') и расч. revenue
    df = chunk_df.with_columns(
        pl.col("date").str.slice(0, 7).alias("month"),
        (pl.col("quantity") * pl.col("price")).alias("revenue")
    )
    grouped = df.group_by(["product", "month"]).agg(
        pl.col("revenue").sum().alias("monthly_revenue")
    )
    result: Dict[str, Dict[str, float]] = {}
    for row in grouped.iter_rows(named=True):
        prod = row["product"]
        month = row["month"]
        revenue = row["monthly_revenue"]
        if prod not in result:
            result[prod] = {}
        result[prod][month] = result[prod].get(month, 0.0) + revenue
    return result

@ray.remote
def xyz_analysis(
    chunks: List[ray.ObjectRef],
    cv_thresholds: Tuple[float, float] = (10.0, 25.0)
) -> Dict[str, List[Tuple[str, float]]]:

    # 1)Параллельно запуск вычисления для каждого чанка
    local_results: List[Dict[str, Dict[str, float]]] = ray.get(
        [xyz_analysis_chunk.remote(c) for c in chunks]
    )

    # 2)Глобальное объединение: суммируем выручку по каждому (product, month)
    global_data: Dict[str, Dict[str, float]] = {}
    for local in local_results:
        for prod, month_dict in local.items():
            if prod not in global_data:
                global_data[prod] = {}
            for month, revenue in month_dict.items():
                global_data[prod][month] = global_data[prod].get(month, 0.0) + revenue

    # 3)Для каждого продукта вычис. коэффициент вариации (CV) месячной выручки
    # CV = (std_dev / mean)
    cats = {"X": [], "Y": [], "Z": []}
    for prod, month_data in global_data.items():
        monthly_revenues = list(month_data.values())
        n = len(monthly_revenues)
        if n == 0:
            continue
        mean_rev = sum(monthly_revenues) / n

        if mean_rev == 0:
            cv = 0.0
        else:
            variance = sum((rev - mean_rev) ** 2 for rev in monthly_revenues) / n
            std_dev = math.sqrt(variance)
            cv = std_dev / mean_rev

        cv_percent = cv * 100.0
        if cv_percent < cv_thresholds[0]:
            cats["X"].append((prod, cv_percent))
        elif cv_percent < cv_thresholds[1]:
            cats["Y"].append((prod, cv_percent))
        else:
            cats["Z"].append((prod, cv_percent))
    return cats


