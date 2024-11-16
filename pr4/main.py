import numpy as np
import matplotlib.pyplot as plt

def is_cell_satisfied(grid, row, col, min_neighbors_for_satisfaction):
    """Проверяет, удовлетворена ли клетка"""
    same_color_neighbors = 0
    cell_value = grid[row, col]
    if cell_value == 0:  # Пустая клетка всегда удовлетворена
        return True

    rows, cols = grid.shape
    # Координаты соседей с учетом краев сетки
    neighbor_positions = [
        (row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
        (row, col - 1), (row, col + 1),
        (row + 1, col - 1), (row + 1, col), (row + 1, col + 1)
    ]
    
    # Подсчет соседей того же цвета
    for n_row, n_col in neighbor_positions:
        if 0 <= n_row < rows and 0 <= n_col < cols and grid[n_row, n_col] == cell_value:
            same_color_neighbors += 1

    return same_color_neighbors >= min_neighbors_for_satisfaction

def initialize_grid(grid_size, blue_ratio, red_ratio):
    """Создает начальную конфигурацию сетки"""
    total_cells = grid_size ** 2
    num_blue = round(total_cells * blue_ratio)
    num_red = round(total_cells * red_ratio)
    
    flat_grid = np.zeros(total_cells)  # Пустые клетки
    flat_grid[:num_blue] = -1  # Синие клетки
    flat_grid[num_blue:num_blue + num_red] = 1  # Красные клетки
    
    np.random.shuffle(flat_grid)
    return flat_grid.reshape((grid_size, grid_size))

def run_schelling_simulation(grid_size=20, blue_ratio=0.45, red_ratio=0.45, min_satisfaction=2, max_iterations=100):
    """Запускает модель Шеллинга с визуализацией процесса"""
    grid = initialize_grid(grid_size, blue_ratio, red_ratio)
    fig, ax = plt.subplots()
    plt.ion()

    for iteration in range(max_iterations):
        satisfaction_map = np.zeros_like(grid, dtype=bool)
        for i in range(grid_size):
            for j in range(grid_size):
                satisfaction_map[i, j] = is_cell_satisfied(grid, i, j, min_satisfaction)

        dissatisfied_cells = np.where((satisfaction_map == False) & (grid != 0))
        vacant_cells = np.where(grid == 0)

        if len(dissatisfied_cells[0]) == 0:
            break  # Все клетки удовлетворены

        random_dissatisfied_index = np.random.randint(len(dissatisfied_cells[0]))
        random_vacant_index = np.random.randint(len(vacant_cells[0]))

        # Перемещаем неудовлетворенную клетку в случайную пустую клетку
        grid[vacant_cells[0][random_vacant_index], vacant_cells[1][random_vacant_index]] = \
            grid[dissatisfied_cells[0][random_dissatisfied_index], dissatisfied_cells[1][random_dissatisfied_index]]
        grid[dissatisfied_cells[0][random_dissatisfied_index], dissatisfied_cells[1][random_dissatisfied_index]] = 0

        ax.clear()
        ax.set_title(f"Итерация {iteration + 1} | k: {min_satisfaction}")
        ax.imshow(grid, cmap='bwr', vmin=-1, vmax=1)
        plt.pause(0.5)

        # Сохранение изображения каждые 10 итераций
        if (iteration + 1) % 10 == 0:
            plt.savefig(f'grid_state_iteration_{iteration + 1}.jpg', format='jpg')

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    grid_size = int(input("Введите размер сетки: "))
    blue_ratio = float(input("Введите долю синих клеток: "))
    red_ratio = float(input("Введите долю красных клеток: "))
    
    if blue_ratio + red_ratio > 0.9:
        raise ValueError("Сумма долей синих и красных клеток должна быть <= 0.9, чтобы оставить место для пустых клеток")
    
    min_satisfaction = int(input("Введите минимальное количество соседей одного цвета для удовлетворения: "))

    run_schelling_simulation(grid_size, blue_ratio, red_ratio, min_satisfaction)