import numpy as np

"""Игра угадай число
Компьютер сам загадывает и сам угадывает число
"""


def random_predict(number: int = 1) -> int:
    """Рандомно угадываем число

    Args:
        number (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Число попыток
    """
    count = 0
    min_ = 0
    max_ = 100

    while True:
        count += 1
        predict_number = round((max_ + min_) / 2)  # предполагаемое число
        if predict_number < number:
            min_ = predict_number

        elif number < predict_number:
            max_ = predict_number

        elif number == predict_number:
            break  # выход из цикла если угадали
    return count


def score_game(random_predict) -> float:
    """За какое количство попыток в среднем за 1000 подходов угадывает наш алгоритм

    Args:
        random_predict ([type]): функция угадывания

    Returns:
        float: среднее количество попыток
    """
    count_ls = []
    #np.random.seed(1)  # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(1000))  # загадали список чисел

    for number in random_array:
        count_ls.append(random_predict(number))

    score = round(np.mean(count_ls), 3)
    print(f"Ваш алгоритм угадывает число в среднем за: {score} попыток")
    return score


if __name__ == "__main__":
    # RUN
    score_game(random_predict)