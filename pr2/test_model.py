import pickle
import main

with open('pr2/trained_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Используйте загруженную модель для предсказаний
test_user = [1, 22, 70, 1, 1, 70, 0, 1, 355, 8, 0, 1, 0, 0, 1]
predicted_drink = main.predict_classification(loaded_model, test_user, k=5)

drink_mapping_reverse = {0: 'чай', 1: 'кофе'}
print(f'Предсказанный напиток для нового пользователя: {drink_mapping_reverse[predicted_drink]}')