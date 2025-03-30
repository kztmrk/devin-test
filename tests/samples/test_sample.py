import unittest
from unittest.mock import patch

import pytest

from tests.samples.sample_module import async_add, process_user_data

###
# pytestのコードのサンプル
# test_**.pyのファイル作成
# test_**やTest**となるようにクラスや関数を宣言する
###


def test_addition():
    # この assertion は失敗します
    # assert 1 + 1 == 3
    assert 1 + 1 == 2


class TestSampleClass:
    def test_method(self):
        x = 10
        y = 5
        # この assertion は失敗します
        # assert x < y
        assert x > y


@pytest.mark.parametrize(
    "input,expected",
    [
        (2, 4),
        (3, 6),
        # # この case は失敗します
        # (4, 9),
    ],
)
def test_multiplication(input, expected):
    assert input * 2 == expected


def test_dictionary_content():
    my_dict = {"a": 1, "b": 2}
    # この assertion は KeyError を発生させます
    # assert my_dict["c"] == 3
    assert my_dict["b"] == 2


class TestSamplePatch(unittest.TestCase):
    @patch("tests.samples.sample_module.get_user_data")
    def test_process_user_data_success(self, mock_get_user_data):
        """
        process_user_data関数が正常に動作する場合のテスト。
        get_user_data関数をモック化して、成功ケースをシミュレートする。
        """
        # モックの戻り値を設定
        mock_get_user_data.return_value = {"name": "John Doe", "age": 30}

        # 関数を呼び出し
        result = process_user_data(1)

        # アサーション
        self.assertEqual(result, "User John Doe is 30 years old")
        mock_get_user_data.assert_called_once_with(1)

    @patch("tests.samples.sample_module.get_user_data")
    def test_process_user_data_not_found(self, mock_get_user_data):
        """
        ユーザーが見つからない場合のprocess_user_data関数の動作をテスト。
        get_user_data関数をモック化して、ユーザーが見つからないケースをシミュレートする。
        """
        # モックの戻り値を設定
        mock_get_user_data.return_value = None

        # 関数を呼び出し
        result = process_user_data(1)

        # アサーション
        self.assertEqual(result, "User not found")
        mock_get_user_data.assert_called_once_with(1)

    # 以下は意図的に失敗させるテストの例。必要に応じてコメントを解除してください。
    # @patch('tests.samples.sample_module.get_user_data')
    # def test_process_user_data_fail(self, mock_get_user_data):
    #     """
    #     意図的に失敗させるテストケース。
    #     実際のアプリケーションでこのようなテストは通常行わないが、
    #     テスト失敗時の動作を確認するために含めている。
    #     """
    #     mock_get_user_data.return_value = {"name": "Jane Doe", "age": 25}
    #     result = process_user_data(1)
    #     self.assertEqual(result, "User Jane Doe is 30 years old")  # 年齢が異なるため失敗
    #     mock_get_user_data.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_async_add_success():
    result = await async_add(2, 3)
    assert result == 5
