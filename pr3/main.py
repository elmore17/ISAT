import vk_api
import json
import time
import os
import aiohttp
import asyncio
from dotenv import load_dotenv
from vk_api.exceptions import ApiError


load_dotenv()
#164679738 id Александр
#419376445 id Марии
#472133870 id Danil
#386272361 id Никита
#'172350665', '229180632', '145195585', '193887357', '386272361', '204720239', '162225997', '860446539', '472133870', '195614586', '825545292', '750743366', '637593527', '299106540', '164679738', '101098087', '239666833', '342040017', '205762499', '165171730', '270780454', '155290829', '151413977', '62269831', '253407490', '192574298', '144399122', '419376445', '508644412', '396854328'
access_token = os.getenv("TOKEN")

vk_session = vk_api.VkApi(token=access_token)
vk = vk_session.get_api()

# Функция для получения информации о пользователе
def get_user_info(user_id):
    try:
        user_info = vk.users.get(user_ids=user_id, fields='photo_50')[0]
        return {
            "id": user_id,
            "first_name": user_info['first_name'],
            "last_name": user_info['last_name'],
            "photo": user_info['photo_50']
        }
    except ApiError as e:
        print(f"Ошибка получения данных пользователя (ID: {user_id}): {e}")
        return None

# Функция для получения списка друзей
def get_friends(user_id):
    try:
        friends = vk.friends.get(user_id=user_id, count=100)
        return friends.get('items', [])
    except ApiError as e:
        print(f"Ошибка получения списка друзей (ID: {user_id}): {e}")
        return []

# Функция для обработки пользователя и записи данных в файл
async def process_user(user_id, file):
    user_info = get_user_info(user_id)
    if not user_info:
        return

    user_data = {
        "id": user_info["id"],
        "photo": user_info["photo"],
        "friends": []
    }

    friends = get_friends(user_id)

    for friend_id in friends[:100]:  # Ограничение на 100 друзей
        friend_info = get_user_info(friend_id)
        if not friend_info:
            continue

        friend_data = {
            "id": friend_id,
            "name": f"{friend_info['first_name']} {friend_info['last_name']}",
            "photo": friend_info['photo'],
            "friends_of_friend": []
        }

        friends_of_friend = get_friends(friend_id)

        for fof_id in friends_of_friend[:100]:  # Ограничение на 100 друзей друзей
            fof_info = get_user_info(fof_id)
            if not fof_info:
                continue

            friend_data["friends_of_friend"].append({
                "id": fof_id,
                "name": f"{fof_info['first_name']} {fof_info['last_name']}",
                "photo": fof_info['photo']
            })

        user_data["friends"].append(friend_data)

    # Запись данных в файл по мере их получения
    file.write(json.dumps({f"{user_info['first_name']} {user_info['last_name']}": user_data}, ensure_ascii=False, indent=4) + ',\n')

async def main():
    team = {'172350665', '229180632', '145195585', '193887357', '386272361', '204720239', '162225997', '860446539', '472133870', '195614586', '825545292', '750743366', '637593527', '299106540', '164679738', '101098087', '239666833', '342040017', '205762499', '165171730', '270780454', '155290829', '151413977', '62269831', '253407490', '192574298', '144399122', '419376445', '508644412', '396854328'}

    with open('team_data1.json', 'w', encoding='utf-8') as file:
        file.write('{"team": {\n')

        tasks = [process_user(user_id, file) for user_id in team]
        await asyncio.gather(*tasks)

        file.write('\n}}')

if __name__ == '__main__':
    asyncio.run(main())