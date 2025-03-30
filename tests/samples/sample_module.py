import asyncio

# ユーザーデータを格納する簡単なデータベースの代わりとなる辞書
USER_DATABASE = {
    1: {"name": "John Doe", "age": 30},
    2: {"name": "Jane Smith", "age": 25},
    3: {"name": "Bob Johnson", "age": 40},
}


def get_user_data(user_id):
    """
    指定されたユーザーIDに基づいてユーザーデータを取得する関数。
    実際のデータベースの代わりに、ローカルの辞書を使用する。

    :param user_id: ユーザーID
    :return: ユーザーデータ（辞書）またはNone（ユーザーが見つからない場合）
    """
    return USER_DATABASE.get(user_id)


def process_user_data(user_id):
    """
    ユーザーデータを取得し、フォーマットされた文字列を返す関数。

    :param user_id: ユーザーID
    :return: フォーマットされたユーザー情報の文字列
    """
    data = get_user_data(user_id)
    if data:
        return f"User {data['name']} is {data['age']} years old"
    else:
        return "User not found"


# テスト対象の非同期関数
async def async_add(x, y):
    await asyncio.sleep(0.1)  # 非同期処理をシミュレート
    return x + y
