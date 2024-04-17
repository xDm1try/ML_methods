import numpy as np




# Средние векторы для классов
mu0 = np.array([5.936, 2.770, 4.260, 1.326])  # Замените mean_0_feature1, mean_0_feature2, ... на средние значения признаков для класса 0
mu1 = np.array([6.588, 2.974, 5.552, 2.026])  # Замените mean_1_feature1, mean_1_feature2, ... на средние значения признаков для класса 1

# Ковариационные матрицы для классов
cov_matrix0 = np.array([[cov_0_11, cov_0_12, ...],
                         [cov_0_21, cov_0_22, ...],
                         ...])  # Замените cov_0_11, cov_0_12, ... на элементы ковариационной матрицы для класса 0

cov_matrix1 = np.array([[cov_1_11, cov_1_12, ...],
                         [cov_1_21, cov_1_22, ...],
                         ...])  # Замените cov_1_11, cov_1_12, ... на элементы ковариационной матрицы для класса 1

# Вектор наблюдения x, для которого нужно вычислить QDF
x = np.array([x_feature1, x_feature2, ...])  # Замените x_feature1, x_feature2, ... на значения признаков для наблюдения x

# Обратные ковариационные матрицы
cov_matrix0_inv = np.linalg.inv(cov_matrix0)
cov_matrix1_inv = np.linalg.inv(cov_matrix1)

# Вычисление квадратичных разделяющих функций
QDF0 = -0.5 * np.dot(np.dot((x - mu0).T, cov_matrix0_inv), (x - mu0)) - 0.5 * np.log(np.linalg.det(cov_matrix0))
QDF1 = -0.5 * np.dot(np.dot((x - mu1).T, cov_matrix1_inv), (x - mu1)) - 0.5 * np.log(np.linalg.det(cov_matrix1))

# Если необходимо, учитывайте априорные вероятности классов P(C0) и P(C1)
# QDF0 += np.log(P(C0))
# QDF1 += np.log(P(C1))

# Принимаем решение на основе значений QDF
if QDF0 > QDF1:
    predicted_class = 0
else:
    predicted_class = 1

print("Предсказанный класс:", predicted_class)
