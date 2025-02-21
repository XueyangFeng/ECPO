import json
import logging
from model.model import OpenAIClient
from .state.dialogue_history import DialogueHistory
from utils import load_config, load_jsonl, load_json
from user_simulator.prompts import recommender_rater_template, policy_rater_template, expression_rater_template, policy_selector_template, ask_recommendation_template, response_to_clarification_template, recommendation_feedback_template, end_conversation_template

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
    def __init__(self, persona: dict, item: dict, openai_client: OpenAIClient, formats: dict, domain: str):
        self.raw_history = []  # 保存所有交互历史的原始数据
        self.dialogue_history = DialogueHistory()  # 保存处理后的对话历史
        self.persona = persona  # 用户 Persona
        self.item = item  # 当前目标物品
        self.openai_client = openai_client  # OpenAI 客户端
        self.domain = domain  # 对话领域
        self.formats = formats  # 格式

    def generate_user_policy(self, dialogue_history: str, last_turn_response: str):
        """生成用户策略"""
        try:
            def _build_policy_prompt():
                prompt = policy_selector_template.format(
                    domain=self.domain,
                    target=json.dumps(self.item, indent=4),
                    Dialogue_history=dialogue_history,
                    last_turn_response=last_turn_response,
                    behaviour=json.dumps(self.persona["Activities"], indent=4)
                )
                return prompt

            policy_prompt = _build_policy_prompt()
            LOGGER.info(f"Policy Prompt:\n{policy_prompt}")  # 记录 prompt

            # 调用 LLM
            response = self.openai_client.get_single_chat_completion(policy_prompt, response_format=self.formats["policy_selector"])
            LOGGER.info(f"Policy Response:\n{response}")  # 记录模型返回

            return response
        except Exception as e:
            LOGGER.error(f"Error generating user policy: {e}")
            raise

    def generate_user_response(self, user_policy, dialogue_history: str, last_turn_response: str):
        """生成用户回复"""
        try:
            def _build_ask_recommendation_prompt():
                prompt = ask_recommendation_template.format(
                    domain=self.domain,
                    Linguistic_Traits=json.dumps(self.persona["Linguistics"], indent=4),
                    target=json.dumps(self.item, indent=4)
                )
                return prompt
            
            def _build_response_to_clarification_prompt():
                """
                Builds a prompt for simulating a user responding to a clarification question.
                """
                prompt = response_to_clarification_template.format(
                    domain=self.domain,
                    Dialogue_history=dialogue_history,
                    last_turn_response=last_turn_response,
                    target=json.dumps(self.item, indent=4),
                    Linguistic_Traits=json.dumps(self.persona["Linguistics"], indent=4)
                )
                return prompt
            
            def _build_feedback_to_recommendation_prompt():
                prompt = recommendation_feedback_template.format(
                    domain=self.domain,
                    Dialogue_history=dialogue_history,
                    last_turn_response=last_turn_response,
                    target=json.dumps(self.item, indent=4),
                    Linguistic_Traits=json.dumps(self.persona["Linguistics"], indent=4),
                    behaviour=json.dumps(self.persona["Activities"], indent=4)
                )
                return prompt
            
            def _build_end_conversation_prompt():
                prompt = end_conversation_template.format(
                    domain=self.domain,
                    Dialogue_history=dialogue_history,
                    last_turn_response=last_turn_response,
                    Linguistic_Traits=json.dumps(self.persona["Linguistics"], indent=4),
                    reason=user_policy["reason"]
                )
                return prompt
            
            if user_policy["policy"] == "ask_recommendation":
                response_prompt = _build_ask_recommendation_prompt()
            elif user_policy["policy"] == "respond_to_clarification":
                response_prompt = _build_response_to_clarification_prompt()
            elif user_policy["policy"] == "provide_feedback_on_recommendation":
                response_prompt = _build_feedback_to_recommendation_prompt()
            elif user_policy["policy"] == "end_conversation":
                response_prompt = _build_end_conversation_prompt()
            else:
                print(user_policy)
                raise ValueError("unexpected user policy")
                

            LOGGER.info(f"Response Prompt:\n{response_prompt}")  # 记录 prompt

            # 调用 LLM
            response = self.openai_client.get_single_chat_completion(response_prompt, response_format=self.formats["responser"])
            LOGGER.info(f"Response:\n{response}")  # 记录模型返回

            return response
        except Exception as e:
            LOGGER.error(f"Error generating user response: {e}")
            raise

    def generate_user_rater(self, last_user_response: str, dialogue_history: str, last_turn_response: str):
        try:
            def _build_recommendation_rater_prompt():
                prompt = recommender_rater_template.format(
                    domain=self.domain,
                    target=json.dumps(self.item, indent=4),
                    last_turn_response=last_turn_response
                )
                return prompt
            
            def _build_policy_rater_prompt():
                prompt = policy_rater_template.format(
                    domain=self.domain,
                    target=json.dumps(self.item, indent=4),
                    Dialogue_history=dialogue_history,
                    last_turn_response=last_turn_response,
                    last_user_response=last_user_response
                )
                return prompt
            
            def _build_expression_rater_prompt():
                prompt = expression_rater_template.format(
                    domain=self.domain,
                    Dialogue_history=dialogue_history,
                    last_turn_response=last_turn_response
                )
                return prompt

            recommendation_rater_prompt = _build_recommendation_rater_prompt()
            policy_rater_prompt = _build_policy_rater_prompt()
            expression_rater_prompt = _build_expression_rater_prompt()

            recommendation_reward = self.openai_client.get_single_chat_completion(recommendation_rater_prompt, response_format=self.formats["recommender_rater"])
            policy_reward = self.openai_client.get_single_chat_completion(
                policy_rater_prompt, response_format=self.formats["policy_rater"]
            )
            expression_reward = self.openai_client.get_single_chat_completion(expression_rater_prompt, response_format=self.formats["expression_rater"])

            return recommendation_reward, policy_reward, expression_reward
        except Exception as e:
            LOGGER.error(f"Error generating reward: {e}")
            raise

    def step(self, dialogue_history: str, last_turn_response: str):
        """执行用户模拟的一步，包括生成策略和回复"""
        try:
            # 直接生成第一句话
            if last_turn_response == "":
                user_policy = {
                    "reason": "begin to conversation",
                    "policy": "ask_recommendation"
                }
                user_response = self.generate_user_response(user_policy=user_policy, dialogue_history=dialogue_history, last_turn_response="")
                return user_response
            else:

                # 生成用户策略
                user_policy = self.generate_user_policy(
                    dialogue_history=dialogue_history,
                    last_turn_response=last_turn_response,
                )

                # 生成用户回复
                user_response = self.generate_user_response(
                    user_policy=json.loads(user_policy),
                    dialogue_history=dialogue_history,
                    last_turn_response=last_turn_response
                )
                
                recommendation_satisfaction, action_satisfaction, expression_satisfaction = self.generate_user_rater(last_user_response=json.loads(user_response)["response"], dialogue_history=dialogue_history, last_turn_response=last_turn_response)

                LOGGER.info(f"Recommendation Satisfaction: {recommendation_satisfaction}")  # 记录评分
                LOGGER.info(f"Action Satisfaction: {action_satisfaction}")  # 记录评分

                return recommendation_satisfaction, action_satisfaction, expression_satisfaction, user_policy, user_response
        except Exception as e:
            LOGGER.error(f"Error during user simulation step: {e}")
            raise


class UserAgentEnv:
    """管理整个用户模拟环境，包含对话历史的管理"""
    def __init__(self, persona_path, user_id, item_id, config_path, format_path, domain, model_type):
        try:
            # 加载用户 persona 数据和物品数据
            self.persona_item = load_jsonl(persona_path)
            self.persona = self.persona_item[user_id]["Persona"]
            self.item = self.persona_item[user_id]["Items"][item_id]

            # 加载配置和 OpenAI 客户端
            self.config = load_config(config_path)[model_type]
            self.openai_client = OpenAIClient(
                base_url=self.config["base_url"],
                api_key=self.config["api_key"],
                model_path=self.config["model_path"],
            )

            # 加载格式
            if model_type.startswith("openai"):
                self.formats = load_json(f"{format_path}/openai_formats.json")
            else:
                self.formats = load_json(f"{format_path}/vllm_formats.json")

            # 初始化对话历史
            self.dialogue_history = DialogueHistory()

            # 初始化用户模拟器
            self.user_simulator = UserSimulator(
                persona=self.persona,
                item=self.item,
                openai_client=self.openai_client,
                formats=self.formats,
                domain=domain,
            )
        except Exception as e:
            LOGGER.error(f"Error initializing UserAgentEnv: {e}")
            raise

    def reset(self, user_id=0, item_id=0):
        """重置环境到初始状态"""
        try:
            # 更新用户和物品状态
            self.persona = self.persona_item[user_id]["Persona"]
            self.item = self.persona_item[user_id]["Items"][item_id]

            # 更新用户模拟器的属性
            self.user_simulator.persona = self.persona
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
            if last_turn_response == "":
                user_response = self.user_simulator.step(dialogue_history=dialogue_history_str, last_turn_response=last_turn_response)
                self.add_user_message(json.loads(user_response)["response"])
                return {
                    "user_response": user_response
                }
            # 执行用户模拟器的交互逻辑，生成用户行为和满意度评分
            else:
                recommendation_satisfaction, action_satisfaction, expression_satisfaction, user_policy, user_response = self.user_simulator.step(
                    dialogue_history=dialogue_history_str,
                    last_turn_response=last_turn_response,
                )
                # 添加生成的用户消息到对话历史
                self.add_user_message(json.loads(user_response)["response"])

                # 返回交互结果
                return {
                    "recommendation_satisfaction": recommendation_satisfaction,
                    "action_satisfaction": action_satisfaction,
                    "expression_satisfaction": expression_satisfaction,
                    "user_policy": user_policy,
                    "user_response": user_response,
                }
        except Exception as e:
            LOGGER.error(f"Error during environment step: {e}")
            raise
