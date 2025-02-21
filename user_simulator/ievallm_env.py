import json
import logging
from model.model import OpenAIClient
from .state.dialogue_history import DialogueHistory
from utils import load_config, load_jsonl, load_json
from user_simulator.prompts import ievallm_template

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_prompts_and_responses.log"),  # 输出到文件
    ]
)

LOGGER = logging.getLogger("UserSimulator")

class UserSimulator:
    """整合用户记忆管理和动作生成逻辑"""
    def __init__(self, item: dict, openai_client: OpenAIClient):
        self.raw_history = []  # 保存所有交互历史的原始数据
        self.dialogue_history = DialogueHistory()  # 保存处理后的对话历史
        self.item = item  # 当前目标物品
        self.openai_client = openai_client  # OpenAI 客户端


    def generate_user_response(self, dialogue_history: str, last_turn_response: str):
        """生成用户回复"""
        try:
            def _build_prompt():
                prompt =ievallm_template.format(
                    Dialogue_history=dialogue_history,
                    last_turn_response=last_turn_response,
                    target=json.dumps(self.item, indent=4),
                )
                return prompt
            response_prompt = _build_prompt()
            # 调用 LLM
            response = self.openai_client.get_single_chat_completion(response_prompt)
            return response
        except Exception as e:
            LOGGER.error(f"Error generating user response: {e}")
            raise


    def step(self, dialogue_history: str, last_turn_response: str):
        """执行用户模拟的一步，包括生成策略和回复"""
        try:
            user_response = self.generate_user_response(dialogue_history=dialogue_history, last_turn_response="")
            return user_response
        except Exception as e:
            LOGGER.error(f"Error during user simulation step: {e}")
            raise


class UserAgentEnv:
    """管理整个用户模拟环境，包含对话历史的管理"""
    def __init__(self, persona_path, user_id, item_id, config_path, format_path, domain, model_type):
        try:
            # 加载用户 persona 数据和物品数据
            self.persona_item = load_jsonl(persona_path)
            self.item = self.persona_item[user_id]["Items"][item_id]

            # 加载配置和 OpenAI 客户端
            self.config = load_config(config_path)[model_type]
            self.openai_client = OpenAIClient(
                base_url=self.config["base_url"],
                api_key=self.config["api_key"],
                model_path=self.config["model_path"],
            )
            # 初始化对话历史
            self.dialogue_history = DialogueHistory()

            # 初始化用户模拟器
            self.user_simulator = UserSimulator(
                item=self.item,
                openai_client=self.openai_client,
            )
        except Exception as e:
            LOGGER.error(f"Error initializing UserAgentEnv: {e}")
            raise

    def reset(self, user_id=0, item_id=0):
        """重置环境到初始状态"""
        try:
            # 更新用户和物品状态
            self.item = self.persona_item[user_id]["Items"][item_id]
            self.user_simulator.item = self.item

            # 清空对话历史
            self.dialogue_history.clear_history()

            LOGGER.info("Environment reset successfully.")
        except Exception as e:
            LOGGER.error(f"Error resetting environment: {e}")
            raise

    def add_user_message(self, message: str):
        """添加用户消息到对话历史"""
        self.dialogue_history.add_user_message(message)
        LOGGER.debug(f"Added user message: {message}")

    def add_assistant_message(self, message: str):
        """添加助手消息到对话历史"""
        self.dialogue_history.add_assistant_message(message)
        LOGGER.debug(f"Added assistant message: {message}")

    def get_dialogue_history(self) -> DialogueHistory:
        """获取格式化的对话历史"""
        return self.dialogue_history
    
    def update_dialogue_history(self, dialogue_history):
        self.dialogue_history = dialogue_history

    def step(self, crs_response=None) -> dict:
        """
        执行模拟中的一步交互。
        
        Args:
            crs_response (dict): 上一轮助手的回复，包含 "content" 字段。

        Returns:
            dict: 包含推荐满意度、动作满意度、用户策略和用户回复。
        """
        try:
            # 获取当前完整对话历史字符串形式
            dialogue_history_str = str(self.dialogue_history)

            # 获取上一轮助手的回复内容（如果有）
            last_turn_response = crs_response if crs_response else ""
            # 记录上一轮助手回复到对话历史（如果有）
            if crs_response:
                self.add_assistant_message(last_turn_response)
                user_response = self.user_simulator.step(dialogue_history=dialogue_history_str, last_turn_response=last_turn_response)
                self.add_user_message(user_response)
                return {
                    "user_response": json.dumps({'response': user_response}),
                }
            if last_turn_response == "":
                user_response = self.user_simulator.step(dialogue_history=dialogue_history_str, last_turn_response=last_turn_response)
                self.add_user_message(user_response)
                return {
                    "user_response": json.dumps({'response': user_response}),
                }
                print(user_response)
        except Exception as e:
            LOGGER.error(f"Error during environment step: {e}")
            raise
