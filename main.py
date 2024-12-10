from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

class Util:
    @staticmethod
    def normalize(img_arr:np.ndarray) -> np.ndarray:
        '''
        Преобразует матрицу к биполярному виду (нормализация)
        '''
        return np.where(img_arr > 127, 1, -1)

    @staticmethod
    def load_img_as_array(img_folder:str) -> np.ndarray:
        '''
        Загружает картинки из заданной папки и нормализует их.
        Возвращает массив, содержащий нормализованные матрицы.
        '''
        arr = []
        for e in os.listdir(img_folder):
            image = Image.open(f'{img_folder}\\{e}')
            image = image.convert('L')
            norm_image = Util.normalize(np.array(image))
            arr.append(norm_image)
        return arr
    
    @staticmethod
    def show_results(results:list) -> None:
        fig, axes = plt.subplots(len(results), 2)

        cmap = plt.cm.gray
        bounds = [-1.5, 0, 1.5]
        norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        for i, e in enumerate(results):
            axes[i][0].imshow(e[0], cmap=cmap, norm=norm)
            axes[i][0].set_title('Test pattern')
            axes[i][0].axis('off')

            axes[i][1].imshow(e[1], cmap=cmap, norm=norm)
            axes[i][1].set_title('Predicted pattern')
            axes[i][1].axis('off')

        plt.tight_layout()
        plt.show()


class HopfieldNetwork:
    def __init__(self, size:int):
        self._size = size
        self._weights = np.zeros((size, size))
    
    def train(self, patterns:list):
        """
        Обучение сети с использованием правила Хебба.
        """
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)  # Преобразуем в вектор-столбец
            self._weights += np.dot(pattern, pattern.T)
        
        # Убираем самоусиление нейронов
        np.fill_diagonal(self._weights, 0)
    
    def predict(self, input_pattern:np.ndarray, max_iterations=10) -> np.ndarray:
        """
        Восстановление паттерна.
        """
        pattern = input_pattern.flatten()
        for _ in range(max_iterations):
            for i in range(self._size):
                # Обновляем состояние i-го нейрона
                raw_sum = np.dot(self._weights[i], pattern)
                pattern[i] = 1 if raw_sum > 0 else -1
        return pattern

if __name__ == '__main__':
    TRAIN_PATH = os.getcwd() + '\\img\\train'
    TEST_PATH = os.getcwd() + '\\img\\test'

    train = Util.load_img_as_array(TRAIN_PATH)
    test = Util.load_img_as_array(TEST_PATH)

    SIZE_X, SIZE_Y = train[0].shape

    net = HopfieldNetwork(SIZE_X * SIZE_Y)
    net.train(train)

    results = []

    for e in test:
        predicted = net.predict(e).reshape(SIZE_X, SIZE_Y)
        results.append((e, predicted))

    Util.show_results(results)