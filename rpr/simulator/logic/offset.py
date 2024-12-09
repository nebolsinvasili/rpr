import numpy as np


def modify_vector(vector: np.ndarray, func, axis: str = "xyz") -> np.ndarray:
    """
    Изменяет значения вектора в зависимости от параметра axis.

    :param vector: numpy array из трех значений [x, y, z].
    :param func: функция, которую нужно применить к указанным компонентам.
    :param axis: строка, определяющая, какие компоненты изменять ('x', 'xy', 'xyz').

    :return: измененный numpy array.
    """
    if not isinstance(vector, np.ndarray) or len(vector) != 3:
        raise ValueError("Входной вектор должен быть numpy массивом из трех элементов.")

    # Проверка value на допустимые значения
    valid_chars = {"x", "y", "z"}
    if not set(axis).issubset(valid_chars):
        raise ValueError(
            "Параметр axis должен содержать только символы 'x', 'y' или 'z'."
        )

    modified_vector = vector.copy()

    if "x" in axis:
        modified_vector[0] = func(modified_vector[0])
    if "y" in axis:
        modified_vector[1] = func(modified_vector[1])
    if "z" in axis:
        modified_vector[2] = func(modified_vector[2])

    return modified_vector


# Изначальная точка в миллиметрах
def offset_point(vector: np.ndarray, limit: list = [1000, 10000], show: bool = False):
    # Смещение точки в микрометрах
    shift_um = np.random.uniform(*limit, vector.size)

    # Конвертируем смещение из микрометров в миллиметры
    shift_mm = shift_um / 1000.0

    # Применяем смещение к координатам
    shifted_vector = vector + shift_mm

    if show:
        print(f"Изначальная точка: {vector} мм")
        print(f"Новая точка после смещения: {shifted_vector} мм")
    return shifted_vector


if __name__ == "__main__":
    for _ in range(1000):
        modify_vector(vector=np.array([1000, 1000, 0]), func=offset_point, axis="xyz")
