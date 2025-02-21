from .base_crs import BaseCRS
from model.model import OpenAIChatClient, OpenAIClient
from utils import load_config, load_json, load_jsonl
from sentence_transformers import SentenceTransformer
from .tools.retriever import Retriever
import json


System_prompt = """
You are a {domain} recommender, and based on the given conversation context, recommend items that the user will like. Follow the instructions below to complete the task:

At the beginning of the conversation, interact with the searcher to understand his/her preferences.

Once you have enough information, generate a response that matches the user's preferences based on the candidate list I provide you.

Recommendations mustz use the full original title retrieved from the database, for example: Recommend "EASARS Wireless Cat Ear Headphones, Pink Gaming Headset Bluetooth 5.0 for Smartphone, Retractable Mic, 50mm Drivers, RGB Lighting Headset with Mic (USB Dongle Not Included)", not a description or shortened version of the title.
"""

User_prompt_template = """
Here is the subsequent list of items:

{item_list}

Here is the conversation context:

{dialogue_history}

Your response:
"""



class RAGCRS(BaseCRS):
    def __init__(self, config_path, emb_model,
                 domain: str,
                 model_type: str, 
                 index_file: str,
                 metadata_file: str,
                 lora: None,
                 target: str,
                 query_num: int = 5
                 ):
        super().__init__(target=target, query_num=query_num)
        self.config =  load_config(config_path)[model_type]
        if lora != None:
            self.openai_client = OpenAIClient(
                base_url=self.config["base_url"],
                api_key=self.config["api_key"],
                model_path=lora,
            )
        else:
            self.openai_client = OpenAIClient(
                base_url=self.config["base_url"],
                api_key=self.config["api_key"],
                model_path=self.config["model_path"]
            )    
        self.domain = domain
        self.interaction_log = {}
        self.scratchpad = ""
        self.step_n = 0
                #self.emb_model = SentenceTransformer(emb_model)
        self.emb_model = emb_model
        self.retriever = Retriever(index_file=index_file, metadata_file=metadata_file, model=self.emb_model)

    def log_interaction(self, prompt: str, response: str):
        
        step_data = self.interaction_log[self.step_n-1] = {}
        try:
            step_data[f"action {self.step_n-1}"] = {
                "input": prompt,
                "output": response
            }

            step_data[f"action {self.step_n-1}"] = {
                "input": prompt,
                "output": response
            }
        except KeyError as e:
            print(f"KeyError: {e} - Step: {self.step_n-1}, Reply count: {self.reply_count}")
            raise
        except TypeError as e:
            print(f"TypeError: {e} - Invalid data encountered for step {self.step_n-1}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def process_input(self, user_input: str):
        """处理用户输入，更新对话状态"""
        self.dialogue_history.add_user_message(json.loads(user_input)['response'])

    def item2text(self, items: list):
        result_lines = [f"{i+1}. {result['Item']}" for i, result in enumerate(items)]
        result_str = "\n".join(result_lines)
        return result_str

    def step(self, user_input: str):
        """处理输入并生成响应"""
        self.process_input(user_input)  # 处理用户输入
        history = str(self.dialogue_history)
        item_list=self.retriever.query(history, self.query_num)
        items = self.item2text(item_list)

        if self.target in items.lower():
            self.recall = True

        ##print(history)
        self.step_n += 1
        #print(items)
        sys_message = System_prompt.format(domain=self.domain)

        user_message = User_prompt_template.format(item_list=items, dialogue_history=history)


        # 生成响应（简单示例）
        response = self.openai_client.get_single_chat_completion(user_message=user_message, sys_prompt=sys_message)

        self.log_interaction(user_message, response)

        self.dialogue_history.add_assistant_message(response)

        return response

    def get_traj(self):
        return self.interaction_log
    
    def get_scratchpad(self):
        return self.scratchpad
 
    def reset(self):
        self.dialogue_history.clear_history()
        self.step_n = 0
        self.recall = False

    def get_traj(self):
        return self.interaction_log