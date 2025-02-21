from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

class OpenAIClient:
    def __init__(self, base_url: str, api_key: str, model_path: str, response_format=None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_path = model_path
        self.response_format = response_format

    def get_single_chat_completion(self, user_message: str, sys_prompt: str = "You are a helpful assistant"):
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_message}]
        # 获取模型的回复
        completion = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            response_format=self.response_format
        )
        return completion.choices[0].message.content

    def get_multi_chat_completions(self, user_messages: list, sys_prompt: str = "You are a helpful assistant"):
        """
        并发获取多条消息的模型回复。

        参数:
            messages (list): 包含用户输入的消息内容的列表。
            sys_prompt (str): 系统提示（默认是 "You are a helpful assistant"）。

        返回:
            list: 多条消息的模型回复列表。
        """
        # 为每个消息准备参数
        def fetch_response(user_message):
            return self.get_single_chat_completion(user_message, sys_prompt)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_response, user_messages))

        return results


class OpenAIChatClient:
    def __init__(self, base_url: str, api_key: str, model_path: str, response_format=None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_path = model_path
        self.response_format = response_format

    def get_single_chat_completion(self, chat_messages: list, sys_prompt: str = "You are a helpful assistant"):
        messages = [{"role": "system", "content": sys_prompt}] + chat_messages
        # 获取模型的回复
        completion = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            response_format=self.response_format
        )
        return completion.choices[0].message.content

    def get_multi_chat_completions(self, chat_messages: list[list], sys_prompt: str = "You are a helpful assistant"):
        """
        并发获取多条消息的模型回复。

        参数:
            messages (list): 包含用户输入的消息内容的列表。
            sys_prompt (str): 系统提示（默认是 "You are a helpful assistant"）。

        返回:
            list: 多条消息的模型回复列表。
        """
        # 为每个消息准备参数
        def fetch_response(user_message):
            return self.get_single_chat_completion(user_message, sys_prompt)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_response, chat_messages))

        return results
