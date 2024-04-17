import numpy as np

def quadratic_discriminant_function(x, class_mean, class_covariance):
    x = x.reshape(-1, 1)  # Преобразуем вектор x в столбец
    class_mean = class_mean.reshape(-1, 1)  # Преобразуем среднее значение класса в столбец
    delta_x = x - class_mean
    inv_covariance = np.linalg.inv(class_covariance)
    
    # Вычисляем значение квадратичной разделяющей функции
    qdf = -0.5 * delta_x.T @ inv_covariance @ delta_x - 0.5 * np.log(np.linalg.det(class_covariance))
    
    return qdf[0, 0]

# Пример использования
# Пусть у вас есть оценки ковариационных матриц и средние значения для двух классов
cov_matrix_class1 = np.array([[1.0, 0.5], [0.5, 2.0]])
cov_matrix_class2 = np.array([[2.0, 0.3], [0.3, 1.0]])

mean_class1 = np.array([1.0, 2.0])
mean_class2 = np.array([2.0, 3.0])

# Пусть у вас есть вектор x, для которого вы хотите вычислить квадратичную разделяющую функцию
x = np.array([1.5, 2.5])

# Вычисляем квадратичную разделяющую функцию для обоих классов
qdf_class1 = quadratic_discriminant_function(x, mean_class1, cov_matrix_class1)
qdf_class2 = quadratic_discriminant_function(x, mean_class2, cov_matrix_class2)

# Теперь у вас есть значения квадратичной разделяющей функции для каждого класса
print("QDF for Class 1:", qdf_class1)
print("QDF for Class 2:", qdf_class2)
print(qdf_class2 - qdf_class1)



# Выбираем класс с наибольшим значением QDF
if qdf_class1 > qdf_class2:
    predicted_class = 1
else:
    predicted_class = 2

print("Predicted Class:", predicted_class)
