from graphviz import Digraph

dot = Digraph(comment='Ray Computation Graph with Iceberg')

# 1) Data Generation
dot.node('G', 'generate_sales_data')
dot.edge('G', 'IS')

# 2) Optional Iceberg branch
dot.node('IS', 'IcebergManager.save_sales')
dot.node('IR', 'IcebergManager.read_sales')
dot.edge('IS', 'IR')

# 3) Parallel Analysis
# 3.1 ABC
dot.node('R', 'product_revenue_chunk')
dot.node('A', 'abc_analysis')
dot.edge('IR', 'R')
dot.edge('R', 'A')

# 3.2 XYZ
dot.node('Xc', 'xyz_analysis_chunk')
dot.node('X', 'xyz_analysis')
dot.edge('IR', 'Xc')
dot.edge('Xc', 'X')

# 4) Reporting
dot.node('Rep', 'generate_report')
dot.node('P', 'PNG & Excel')
dot.edge('A', 'Rep')
dot.edge('X', 'Rep')
dot.edge('Rep', 'P')

# Рендеринг
dot.render('ray_computation_graph', format='svg', cleanup=True)


