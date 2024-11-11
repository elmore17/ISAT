import json
import aiohttp
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

# Загрузка данных команды
with open('pr3/team_data.json', 'r', encoding='utf-8') as file:
    team_data = json.load(file)

access_token = os.getenv("TOKEN")
friendship_cache = {}

async def are_friends(session, user_id, friend_id):
    """Асинхронно проверяет, являются ли два пользователя друзьями, используя кэш."""
    if (user_id, friend_id) in friendship_cache:
        return friendship_cache[(user_id, friend_id)]
    if (friend_id, user_id) in friendship_cache:
        return friendship_cache[(friend_id, user_id)]

    url = f"https://api.vk.com/method/friends.areFriends"
    params = {
        "user_ids": f"{user_id},{friend_id}",
        "access_token": access_token,
        "v": "5.131"
    }
    
    try:
        async with session.get(url, params=params) as response:
            data = await response.json()
            if "response" in data:
                is_friend = any(friendship['user_id'] == friend_id and friendship['friend_status'] == 3 for friendship in data['response'])
                friendship_cache[(user_id, friend_id)] = is_friend
                return is_friend
    except Exception as e:
        print(f"Ошибка при запросе VK API: {e}")
        return False

async def update_and_save_structure(team):
    """Создает структуру с проверкой дружбы внутри friends_of_friend и сохраняет JSON после проверки каждого friend_of_friend."""
    new_team_structure = {'team': {}}
    async with aiohttp.ClientSession() as session:
        for user_name, user in team['team'].items():
            user_data = {
                'id': user.get('id'),
                'photo': user.get('photo'),
                'friends': []
            }
            
            for friend in user.get('friends', []):
                friend_data = {
                    'id': friend.get('id'),
                    'name': friend.get('name'),
                    'photo': friend.get('photo'),
                    'friends_of_friend': []
                }
                
                for friend_of_friend in friend.get('friends_of_friend', []):
                    friend_of_friend_data = {
                        'id': friend_of_friend.get('id'),
                        'name': friend_of_friend.get('name'),
                        'photo': friend_of_friend.get('photo'),
                        'friend_with': []
                    }
                    
                    tasks = []
                    other_friends = []
                    
                    # Создаем задачи для проверки дружбы
                    for other_friend_of_friend in friend.get('friends_of_friend', []):
                        if friend_of_friend['id'] != other_friend_of_friend['id']:
                            tasks.append(are_friends(session, friend_of_friend['id'], other_friend_of_friend['id']))
                            other_friends.append(other_friend_of_friend)
                    
                    # Выполняем все проверки дружбы одновременно
                    results = await asyncio.gather(*tasks)
                    
                    # Обрабатываем результаты дружбы, синхронизировано с other_friends
                    for idx, is_friend in enumerate(results):
                        if is_friend:
                            friend_of_friend_data['friend_with'].append(other_friends[idx]['id'])
                    
                    friend_data['friends_of_friend'].append(friend_of_friend_data)
                
                user_data['friends'].append(friend_data)
                
                # Сохраняем промежуточные данные после каждого друга
                new_team_structure['team'][user_name] = user_data
                with open('pr3/team_data_with_friends.json', 'w', encoding='utf-8') as outfile:
                    json.dump(new_team_structure, outfile, indent=4, ensure_ascii=False)
        
            new_team_structure['team'][user_name] = user_data
    
    return new_team_structure

asyncio.run(update_and_save_structure(team_data))