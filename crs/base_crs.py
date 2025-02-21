from abc import ABC, abstractmethod
from utils import load_json
from .dialogue_history import DialogueHistory

class BaseCRS(ABC):
    def __init__(self, target: str, query_num: int = 5):
        """
        初始化基础对话推荐系统
        :param dialogue_state: 对话状态对象
        """
        self.dialogue_history = DialogueHistory()  # 对话历史
        self.scratchpad = "" #规划历史
        self.target = target
        self.recall = False
        self.query_num = query_num

    @abstractmethod
    def process_input(self, user_input: str):
        """处理用户输入，解析意图和相关信息"""
        pass

    @abstractmethod
    def step(self, user_input: str):
        """每个子类实现自己的步骤逻辑，执行任务"""
        pass

    @abstractmethod  
    def reset(self):
        """重置对话历史和对话状态"""
        self.dialogue_history.clear()

    def get_traj(self):
        return self.scratchpad


    def update_history(self, user_input: str, system_response: str):
        """更新对话历史"""
        self.dialogue_history.append({'user': user_input, 'system': system_response})

    
    def get_recall(self):
        return self.recall