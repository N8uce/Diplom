import matplotlib.pyplot as plt

# Данные
cores = [1, 2, 3, 4, 5, 6]

# Для 2.5 млн записей
abc_xyz_time_2_5m = [2.6382, 1.4464, 1.3421, 0.8110, 0.6501, 0.5773]
total_time_2_5m = [65.8533, 39.3448, 31.2887, 25.3690, 23.0005, 20.7694]

# Для 5 млн записей
abc_xyz_time_5m = [3.7744, 2.0687, 1.7457, 1.6345, 1.5797, 1.1994]
total_time_5m = [123.7812, 75.5707, 50.1840, 42.3801, 37.3735, 36.6352]

# Построение графиков
plt.figure(figsize=(12, 6))

plt.plot(cores, total_time_2_5m, marker='o', label='2.5 млн записей')
plt.plot(cores, total_time_5m, marker='o', label='5 млн записей')
plt.title('Общее время выполнения')
plt.xlabel('Количество ядер')
plt.ylabel('Время (сек.)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("abc_xyz_performance.png")
