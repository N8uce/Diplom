#benchmark.py
import ray
import time
import argparse
from analysis import abc_analysis, xyz_analysis
from iceberg_manager import IcebergManager
from data_generation import generate_sales_data
import os
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl

def generate_report(abc_res, xyz_res):
    # 1)ABC: печать продуктов по категориям
    print("\nABC Категории:")
    for cat in ("A", "B", "C"):
        prods = abc_res.get(cat, [])
        print(f"- Категории {cat} ({len(prods)} продукты):")
        for p in sorted(prods):
            print(f"    • {p}")

    # 2)XYZ: печать продуктов по категориям
    print("\nXYZ Категории:")
    for cat in ("X", "Y", "Z"):
        items = xyz_res.get(cat, [])
        print(f"- Категории {cat} ({len(items)} продукты):")
        for p, cv in sorted(items):
            print(f"    • {p} (CV={cv:.2f}%)")

    # 3)Подготовка данных для графиков
    abc_counts = {cat: len(abc_res.get(cat, [])) for cat in ("A", "B", "C")}
    xyz_counts = {cat: len(xyz_res.get(cat, [])) for cat in ("X", "Y", "Z")}

    # 4)Построение и сохранение бар-чартов
    import matplotlib.pyplot as plt

    plt.figure()
    plt.bar(abc_counts.keys(), abc_counts.values())
    plt.title('Распределение по категориям ABC')
    plt.xlabel('Категория')
    plt.ylabel('Количество продуктов')
    abc_img = 'abc_report.png'
    plt.savefig(abc_img)
    print(f"\nSaved ABC report chart: {abc_img}")
    plt.close()

    plt.figure()
    plt.bar(xyz_counts.keys(), xyz_counts.values())
    plt.title('Распределение по категориям XYZ')
    plt.xlabel('Категория')
    plt.ylabel('Количество продуктов')
    xyz_img = 'xyz_report.png'
    plt.savefig(xyz_img)
    print(f"Saved XYZ report chart: {xyz_img}\n")
    plt.close()

    # 5)Построение матрицы пересечения ABC-XYZ
    from collections import defaultdict
    import seaborn as sns
    import numpy as np

    abc_map = {}
    for cat in abc_res:
        for p in abc_res[cat]:
            abc_map[p] = cat

    xyz_map = {}
    for cat in xyz_res:
        for p, _ in xyz_res[cat]:
            xyz_map[p] = cat

    #Подсчёт пересечений и сохранение списка товаров
    matrix = defaultdict(list)
    for p in abc_map:
        if p in xyz_map:
            key = (abc_map[p], xyz_map[p])
            matrix[key].append(p)

    #Матрица 3x3: A,B,C x X,Y,Z
    abc_labels = ["A", "B", "C"]
    xyz_labels = ["X", "Y", "Z"]
    data = np.zeros((3, 3), dtype=int)

    for i, a in enumerate(abc_labels):
        for j, x in enumerate(xyz_labels):
            data[i, j] = len(matrix.get((a, x), []))

    plt.figure(figsize=(6, 5))
    sns.heatmap(data, annot=True, fmt="d", xticklabels=xyz_labels, yticklabels=abc_labels, cmap="Blues")
    plt.title("Пересечение категорий ABC–XYZ")
    plt.xlabel("Категория XYZ")
    plt.ylabel("Категория ABC")
    cross_img = 'abc_xyz_cross.png'
    plt.savefig(cross_img)
    print(f"Saved ABC-XYZ cross matrix chart: {cross_img}\n")
    plt.close()

    print("\nДетализация пересечения ABC–XYZ:")
    for a in abc_labels:
        for x in xyz_labels:
            items = matrix.get((a, x), [])
            if items:
                print(f"Категория {a}{x} ({len(items)}): {', '.join(sorted(items))}")

    import pandas as pd

    # Сохранение ABC-анализ в Excel
    abc_data = [(cat, prod) for cat, prods in abc_res.items() for prod in prods]
    df_abc = pd.DataFrame(abc_data, columns=["ABC Категория", "Продукт"])
    df_abc.to_excel("abc_report.xlsx", index=False)

    # Сохранение XYZ-анализ в Excel
    xyz_data = [(cat, prod, round(cv, 2)) for cat, items in xyz_res.items() for prod, cv in items]
    df_xyz = pd.DataFrame(xyz_data, columns=["XYZ Категория", "Продукт", "CV (%)"])
    df_xyz.to_excel("xyz_report.xlsx", index=False)

    # Сохранение ABC-XYZ пересечения
    cross_data = [(a, x, ", ".join(sorted(matrix[(a, x)]))) for a in abc_labels for x in xyz_labels if
                  matrix.get((a, x))]
    df_cross = pd.DataFrame(cross_data, columns=["ABC", "XYZ", "Продукты"])
    df_cross.to_excel("abc_xyz_cross.xlsx", index=False)

    print("Сохранены Excel-файлы: abc_report.xlsx, xyz_report.xlsx, abc_xyz_cross.xlsx")


def run_benchmark(num_cpus: int, data_size: int, batch_size: int, iceberg: bool=False,
                  catalog_type: str="rest-minio", report: bool=False):
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    @ray.remote
    def _warmup():
        return 0
    ray.get([_warmup.remote() for _ in range(num_cpus)])

    num_chunks = num_cpus

    forced_xyz = {
        "Camera": "Z",
        "Laptop": "X",
        "Tablet": "Y",
    }
    forced_abc = {
        #A
        "TV": "A", "Smartwatch": "A","Camera": "A",
        #B
        "Monitor": "C", "Speaker": "B", "Tablet": "C",
        "Drone": "C"
    }

    chunk_refs, gen_time = generate_sales_data(data_size, num_chunks,forced_xyz=forced_xyz,forced_abc=forced_abc)

    #chunks = ray.get(chunk_refs)

    #!!!!!!!
    iceberg_mgr = None
    if iceberg:
        iceberg_mgr = IcebergManager(catalog_type=catalog_type)


        full_df = pl.concat(ray.get(chunk_refs), rechunk=True)
        iceberg_mgr.save_sales(full_df)


        stored_df = iceberg_mgr.read_sales()
        print(f"Read {len(stored_df)} rows from Iceberg")
        datalen = len(stored_df)

        chunk_size = len(stored_df) // num_chunks
        chunks = [stored_df[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

        if len(stored_df) % num_chunks != 0:
            chunks[-1] = pl.concat([chunks[-1], stored_df[num_chunks * chunk_size:]])

        chunk_refs = [ray.put(chunk) for chunk in chunks]

    # dispatch both at once
    abc_ref = abc_analysis.remote(chunk_refs)
    xyz_ref = xyz_analysis.remote(chunk_refs)

    # now block for both and time each “from dispatch to result”
    start = time.time()
    abc_res, xyz_res = ray.get([abc_ref, xyz_ref])
    total = time.time() - start

    if report:
        generate_report(abc_res, xyz_res)

    ray.shutdown()

    if iceberg_mgr:
        iceberg_mgr.cleanup()

    return {
        "num_cpus": num_cpus,
        "data_size": data_size,
        "batch_size": batch_size,
        "datalen": datalen,
        "totalabc-xyz": total
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-size", type=int, default=100000,
                        help="Total number of generated sales records")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for ABC analysis")
    parser.add_argument("--iceberg", action="store_true",
                        help="Use Iceberg-backed storage for data")
    parser.add_argument("--report", action="store_true",
                        help="Generate detailed report with category assignments and charts")
    parser.add_argument(
        "--catalog-type",
        choices=["rest-minio", "sql"],
        default="rest-minio",
        help="Type of Iceberg catalog: 'rest-minio', 'sql'"
    )
    parser.add_argument(
        "--cpus", type=int, nargs='+', default=[1, 2, 4, 6],
        help="List of CPU counts to benchmark, e.g., --cpus 1 2 4 6"
    )
    args = parser.parse_args()

    results = []
    for cpus in args.cpus:
        print(f"\nRunning benchmark with {cpus} CPUs...")
        start_time = time.time()
        res = run_benchmark(
            cpus,
            args.data_size,
            args.batch_size,
            iceberg=args.iceberg,
            catalog_type=args.catalog_type,
            report=args.report
        )
        elapsed = time.time() - start_time
        res['elapsed_time'] = elapsed
        results.append((cpus, res))
        print(f"Result for {cpus} CPUs: {res}")

    print("\nИтоги:")
    header = f"{'Кол-во Ядер':>5} | {'ABC-XYZ время':>12} | {'Общее время':>12} | {'Объём данных':>13}"
    print(header)
    print("-" * len(header))
    for cpus, r in results:
        print(f"{cpus:>5}       | {r['totalabc-xyz']:>12.4f}  | {r['elapsed_time']:>12.4f} | {r['datalen']:>13}")

if __name__ == "__main__":
    main()
