import pandas as pd
import math
from collections import Counter


# Функция для преобразования времени (часы и минуты) в количество минут с начала суток
def timestamp_to_minutes(timestamp):
    # Обрабатываем строки вида "Час:Минуты"
    if pd.isna(timestamp):
        return None
    return timestamp.hour * 60 + timestamp.minute


def load_data_from_excel(file_path):
    # Загружаем данные
    df = pd.read_excel(file_path)
    
    # Оставляем нужные колонки
    df = df[['Укажите Ваш пол', 'Какой напиток Вы предпочитаете утром?', 'Укажите Ваш возраст', 
             'Насколько здоровый образ жизни Вы ведете? Укажите число по шкале от 0 до 100', 
             'Курите ли Вы?','Укажите цвет Вашего левого глаза', 'Много ли Вы испытываете стресса в жизни? Укажите число от 0 до 100',
             'Вы высыпаетесь?', 'Укажите Ваш хронотип', 'Во сколько Вы обычно просыпаетесь? Укажите время в формате "Час" и "Минуты"',
             'Сколько Вы в среднем спите? Укажите среди время вашего сна', 'Если поблизости с Вашим домом кофейня?', 
             'Вы работаете из офиса?', 'Вы домосед?', 'У Вас есть хронические заболевания?', 
             'Какой рукой Вы пишите?']]
    #Выкинули "знаки зодиака" тк с ними очень низкая точнось
    # Преобразование категориальных данных в числовые
    gender_mapping = {'Мужчина': 0, 'Женщина': 1}
    drink_mapping = {'Чай': 0, 'Кофе': 1}
    smoking_mapping = {'Да': 1, 'Нет': 0}
    eye_color = {'Серо-зеленый': 0, 'Зелёный': 1, 'Голубой': 2, 'Карий': 3, 'Серый': 4, 'Серо-голубой': 5 }
    sleep_quality_mapping = {'Да': 1, 'Нет': 0}
    chronotype_mapping = {'Жаворонок': 0, 'Сова': 1}
    coffee_shop_mapping = {'Да': 1, 'Нет': 0}
    gourmet_mapping = {'Да': 1, 'Нет': 0}
    office_worker_mapping = {'Да': 1, 'Нет': 0}
    homebody_mapping = {'Да': 1, 'Нет': 0}
    chronic_diseases_mapping = {'Да': 1, 'Нет': 0}
    writing_hand_mapping = {'Левой': 0, 'Правой': 1}
    zodiac_mapping = {
        'Овен': 0, 'Телец': 1, 'Близнецы': 2, 'Рак': 3, 'Лев': 4, 'Дева': 5, 'Весы': 6, 
        'Скорпион': 7, 'Стрелец': 8, 'Козерог': 9, 'Водолей': 10, 'Рыбы': 11
    }
    
    # Применение маппингов
    df['Укажите Ваш пол'] = df['Укажите Ваш пол'].map(gender_mapping)
    df['Какой напиток Вы предпочитаете утром?'] = df['Какой напиток Вы предпочитаете утром?'].map(drink_mapping)
    df['Курите ли Вы?'] = df['Курите ли Вы?'].map(smoking_mapping)
    df['Укажите цвет Вашего левого глаза'] = df['Укажите цвет Вашего левого глаза'].map(eye_color)
    df['Вы высыпаетесь?'] = df['Вы высыпаетесь?'].map(sleep_quality_mapping)
    df['Укажите Ваш хронотип'] = df['Укажите Ваш хронотип'].map(chronotype_mapping)
    df['Если поблизости с Вашим домом кофейня?'] = df['Если поблизости с Вашим домом кофейня?'].map(coffee_shop_mapping)
    df['Вы работаете из офиса?'] = df['Вы работаете из офиса?'].map(office_worker_mapping)
    df['Вы домосед?'] = df['Вы домосед?'].map(homebody_mapping)
    df['У Вас есть хронические заболевания?'] = df['У Вас есть хронические заболевания?'].map(chronic_diseases_mapping)
    df['Какой рукой Вы пишите?'] = df['Какой рукой Вы пишите?'].map(writing_hand_mapping)
    df['Во сколько Вы обычно просыпаетесь? Укажите время в формате "Час" и "Минуты"'] = pd.to_datetime(df['Во сколько Вы обычно просыпаетесь? Укажите время в формате "Час" и "Минуты"'], format="%d.%m.%Y %H:%M:%S")
    df['Во сколько Вы обычно просыпаетесь? Укажите время в формате "Час" и "Минуты"'] = df['Во сколько Вы обычно просыпаетесь? Укажите время в формате "Час" и "Минуты"'].apply(timestamp_to_minutes)
    
    # Преобразуем DataFrame в список списков
    dataset = df.values.tolist()
    
    return dataset


# Функция для расчета евклидова расстояния между двумя точками
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):  # исключаем последний элемент, т.к. это метка класса
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

# Функция для поиска k ближайших соседей
def get_neighbors(train, test_row, k):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda x: x[1])  # сортируем по расстоянию
    neighbors = [distances[i][0] for i in range(k)]  # выбираем k ближайших
    return neighbors

# Функция для предсказания класса на основе голосования соседей
def predict_classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_values = [row[-1] for row in neighbors]  # метки классов соседей
    prediction = Counter(output_values).most_common(1)[0][0]  # класс с наибольшим количеством голосов
    return prediction

# Оценка точности
def accuracy_metric(actual, predicted):
    correct = sum(a == p for a, p in zip(actual, predicted))
    return correct / len(actual) if len(actual) > 0 else 0

# Разделим данные на тренировочные и тестовые
def train_test_split(dataset, split_ratio=0.8):
    train_size = int(len(dataset) * split_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]
    return train_set, test_set

def min_max_normalize(dataset):
    dataset_transposed = list(zip(*dataset))
    
    normalized_data = []
    
    for i in range(len(dataset_transposed) - 1):  # пропускаем последний столбец, это метка класса
        column = dataset_transposed[i]
        col_min = min(column)
        col_max = max(column)
        normalized_column = [(x - col_min) / (col_max - col_min) if col_max != col_min else 0 for x in column]
        normalized_data.append(normalized_column)

    normalized_data.append(dataset_transposed[-1])
    normalized_data = list(zip(*normalized_data))
    normalized_data = [list(row) for row in normalized_data]
    
    return normalized_data

file_path = 'pr2/result_the_survey.xlsx'
dataset = load_data_from_excel(file_path)
normalize = min_max_normalize(dataset)

train_set, test_set = train_test_split(normalize)

# Прогнозируем для тестовой выборки
predictions = []
for row in test_set:
    prediction = predict_classification(train_set, row, k=1)
    predictions.append(prediction)

# Фактические классы тестовой выборки
actual = [row[1] for row in test_set]

def analyze_k(train_set, test_set, actual, k_values):
    accuracy_results = {}

    for k in k_values:
        predictions = []
        for row in test_set:
            prediction = predict_classification(train_set, row, k)
            predictions.append(prediction)

        # Рассчитываем точность для данного k
        accuracy = accuracy_metric(actual, predictions)
        accuracy_results[k] = accuracy
        print(f'k={k}, Точность: {accuracy:.2f}')

    return accuracy_results

k_values = range(1, 21)

accuracy_results = analyze_k(train_set, test_set, actual, k_values)

# Находим лучшее k
best_k = max(accuracy_results, key=accuracy_results.get)
print(f'Лучшее значение k: {best_k}, с точностью: {accuracy_results[best_k]:.2f}')

# Оценим точность
accuracy = accuracy_metric(actual, predictions)
print(f'Точность: {accuracy}')

#Тест
test_user = [1,22,70,1,1,70,0,1,355,8,0,1,0,0,1]
predicted_drink = predict_classification(train_set, test_user, k=5)
drink_mapping_reverse = {0: 'чай', 1: 'кофе'}
print(f'Предсказанный напиток для нового пользователя: {drink_mapping_reverse[predicted_drink]}')


#Сборка модели 
import pickle
with open('trained_model.pkl', 'wb') as model_file:
    pickle.dump(train_set, model_file)