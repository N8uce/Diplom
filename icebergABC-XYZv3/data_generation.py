import ray
import random
import numpy as np
from datetime import datetime, timedelta
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl
from typing import Tuple, List, Dict
import time

@ray.remote
def generate_sales_chunk(
    chunk_size: int,
    start_id: int,
    forced_xyz: Dict[str, str],
    forced_abc: Dict[str, str]
) -> pl.DataFrame:

    # 0)Задание ABC-категорий и множителей для классического распределения
    abc_multiplier = {"A": 10.0, "B": 3.0, "C": 1.0}

    # 1)Параметры товаров
    products = {
        "Laptop": {"base_qty": 10, "qty_std": 1,   "trend": 0.1},
        "Phone": {"base_qty": 15,  "qty_std": 2,   "trend": 0.05},
        "Camera": {"base_qty": 50, "qty_std": 65,  "trend": 0.1},
        "Tablet": {"base_qty": 20, "qty_std": 8,   "trend": -0.05},
        "Smartwatch": {"base_qty": 100, "qty_std": 40, "trend": 0.1},
        "Headphones": {"base_qty": 50, "qty_std": 20, "trend": 0.1},
        "Speaker": {"base_qty": 80, "qty_std": 120,"trend": 0.05},
        "Charger": {"base_qty": 5,  "qty_std": 10,  "trend": 0.0},
        "USB Cable": {"base_qty": 100, "qty_std": 30, "trend": 0.1},
        "TV": {"base_qty": 30,   "qty_std": 15,  "trend": 0.07},
        "Gaming Console": {"base_qty": 8, "qty_std": 3, "trend": 0.12},
        "Drone": {"base_qty": 5,  "qty_std": 2,   "trend": 0.15},
        "VRHeadset": {"base_qty": 12, "qty_std": 4, "trend": 0.2},
        "Smart Home Hub": {"base_qty": 20, "qty_std": 5, "trend": 0.05},
        "Monitor": {"base_qty": 25, "qty_std": 10, "trend": 0.03},
        "Keyboard": {"base_qty": 70, "qty_std": 15, "trend": 0.0},
        "Mouse": {"base_qty": 80,   "qty_std": 20, "trend": 0.05},
        "Printer": {"base_qty": 10, "qty_std": 6,   "trend": -0.02},
        "Router": {"base_qty": 15,  "qty_std": 5,   "trend": 0.1},
    }
    # 2) Базовые цены
    base_price_map = {
        "Phone": 200.0, "Laptop": 1500.0, "Tablet": 300.0, "Headphones": 150.0,
        "Smartwatch": 250.0, "Camera": 500.0, "Speaker": 120.0, "Charger": 20.0,
        "USB Cable": 10.0, "TV": 600.0, "Gaming Console": 500.0, "Drone": 800.0,
        "VRHeadset": 400.0, "Smart Home Hub": 150.0, "Monitor": 250.0,
        "Keyboard": 50.0, "Mouse": 40.0, "Printer": 200.0, "Router": 100.0
    }

    customers = ["Alice", "Bob", "Charlie", "Diana"]
    stores    = ["Store1", "Store2", "Store3", "Store4"]
    base_date = datetime(2025, 1, 1)

    # 3) Генерация сезонных весов по XYZ, как было ранее
    product_month_weights: Dict[str, List[float]] = {}
    months = list(range(1, 13))
    for prod_name in products:
        cat = forced_xyz.get(prod_name)
        if cat == "X":
            product_month_weights[prod_name] = [1.0] * 12
        elif cat == "Y":
            w = np.random.normal(loc=1.0, scale=0.3, size=12)
            product_month_weights[prod_name] = np.clip(w, 0.5, 1.5).tolist()
        elif cat == "Z":
            hot = random.sample(range(12), 2)
            product_month_weights[prod_name] = [20.0 if m in hot else 1.0 for m in range(12)]
        else:
            w = np.random.normal(loc=5.0, scale=3.0, size=12)
            product_month_weights[prod_name] = np.clip(w, 0.5, None).tolist()

    # 4) Списки для сбора данных
    ids, prods, custs, dates, stors, quantities, prices = ([] for _ in range(7))

    for i in range(chunk_size):
        pid = start_id + i
        prod = random.choice(list(products.keys()))
        cfg  = products[prod]

        # выбор месяца по весам
        mw = product_month_weights[prod]
        m  = random.choices(months, weights=mw, k=1)[0]
        date = base_date.replace(month=m, day=1) + timedelta(days=random.randint(0, 27))
        days_offset = (date - base_date).days

        # тренд и сезонность
        trend_factor    = 1 + cfg["trend"] * (days_offset / 365)
        seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * days_offset / 365)

        # спайк и буст
        spike_chance    = 0.8
        spike_magnitude = cfg["base_qty"] * random.uniform(10, 40)
        monthly_boost   = random.choice([0.5, 1.0, 1.5, 2.0])

        raw_qty = np.random.normal(cfg["base_qty"] * trend_factor * seasonal_factor,
                                   cfg["qty_std"])
        spike   = (random.random() < spike_chance) * spike_magnitude
        base_qty_calc = raw_qty + spike

        # 5) Применяем ABC-множитель
        abc_cls = forced_abc.get(prod, random.choices(["A", "B", "C"], weights=[0.2, 0.3, 0.5])[0])
        mult = abc_multiplier[abc_cls]
        qty  = max(1, int(base_qty_calc * monthly_boost * mult))

        price = round(random.uniform(0.8, 1.3) * base_price_map[prod], 2)

        # собираем строки
        ids.append(pid)
        prods.append(prod)
        custs.append(random.choice(customers))
        dates.append(date.strftime("%Y-%m-%d"))
        stors.append(random.choice(stores))
        quantities.append(qty)
        prices.append(price)

    # 6) Возвращаем DataFrame
    return pl.DataFrame({
        "id":       ids,
        "product":  prods,
        "customer": custs,
        "date":     dates,
        "store":    stors,
        "quantity": quantities,
        "price":    prices,
    })


def generate_sales_data(
    total_size: int,
    num_cpus: int,
    forced_xyz: Dict[str, str],
    forced_abc: Dict[str, str]
) -> Tuple[List[ray.ObjectRef], float]:
    base, rem = divmod(total_size, num_cpus)
    sizes = [base + 1 if i < rem else base for i in range(num_cpus)]

    start = time.time()
    refs = []
    offset = 0
    for sz in sizes:
        refs.append(generate_sales_chunk.remote(sz, offset, forced_xyz,forced_abc))
        offset += sz
    return refs, time.time() - start
