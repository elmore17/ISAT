import vk_api
import json
import time
import os
from dotenv import load_dotenv


load_dotenv()
#164679738 id Александр
#419376445 id Марии
#472133870 id Danil
#386272361 id Никита

def main():
    team = {'Александр': '164679738', 'Мария': '419376445', 'Данила': '472133870','Никита': '386272361'}
    access_token = os.getenv("TOKEN")
    vk_session = vk_api.VkApi(token=access_token)
    vk = vk_session.get_api()
    team_data = {}

    for name, user_id in team.items():
        try:
            time.sleep(1)
            user_info = vk.users.get(user_ids=user_id, fields='photo_50')[0]
            first_name = user_info['first_name']
            last_name = user_info['last_name']
            photo = user_info['photo_50']

            user_data = {
                "id": user_id,
                "photo": photo,
                "friends": []
            }
            time.sleep(1)
            # Получаем список друзей пользователя
            friends = vk.friends.get(user_id=user_id)['items']
            
            for friend_id in friends:
                try:
                    time.sleep(1)
                    friend_info = vk.users.get(user_ids=friend_id, fields='photo_50')[0]
                    friend_name = f"{friend_info['first_name']} {friend_info['last_name']}"
                    friend_photo = friend_info['photo_50']

                    user_data["friends"].append({
                        "id": friend_id,
                        "name": friend_name,
                        "photo": friend_photo
                    })

                except vk_api.exceptions.ApiError as e:
                    if e.code == 30:
                        print(f"Профиль друга (ID: {friend_id}) закрыт.")
                
                team_data[f"{first_name} {last_name}"] = user_data
        
        except vk_api.exceptions.ApiError as e:
                if e.code == 30:
                    print(f"Профиль друга (ID: {friend_id}) закрыт.")

    with open('team_data.json', 'w', encoding='utf-8') as f:
        json.dump({"team": team_data}, f, ensure_ascii=False, indent=4)

    print("Данные успешно записаны в team_data.json")

if __name__ == '__main__':
    main()
