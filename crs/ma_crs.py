from .base_crs import BaseCRS
from model.model import OpenAIClient
from utils import load_config, load_json, load_jsonl
from .tools.retriever import Retriever
import json
import re
from sentence_transformers import SentenceTransformer

Ask_prompt_template="""
You are a {Domain} conversational recommender system. Generate questions to further refine the user's preferences or clarify their needs based on the current profile, context, and past reflections.

- Input:
  - User Profile: {USER_PROFILE}
  - Dialogue History: {DIALOGUE_HISTORY}
  - Reflection: {REFLECTION}

- Objectives:
  1. Ask simple, direct questions to fill in gaps in the user's profile. Avoid asking overly detailed questions that could overwhelm the user.
  2. Use past reflections to avoid repetitive questions or clarify why certain questions were ineffective.
  3. If recommendations fail or the user provides negative feedback, ask clarifying questions to redirect the conversation.
  4. Ensure questions are concise and contextually relevant, helping the user gradually elaborate on their needs without overwhelming them with too many details at once.

Please create your inquiry response:
"""

Act_prompt_template="""
You are a {Domain} conversational recommender system. Generate a structured and semantically meaningful query based on the user's input, profile context, and past reflections. This query will be used to retrieve relevant items or information from external systems or databases.

- Input:
  - Dialogue History: {DIALOGUE_HISTORY}
  - User Profile: {USER_PROFILE}
  - Reflection: {REFLECTION}

- Objectives:
  1. Analyze user input and previous interactions to identify key entities, preferences, and intents.
  2. Use past reflections to improve query specificity, precision, and alignment with user needs.
  3. Combine this analysis with user profile information to construct a complete query.

Please output the query:
"""

Rec_prompt_template_v0="""
You are a {Domain} conversational recommender system. Generate concise and personalized recommendations for the user based on their profile, dialogue context, retrieved data, and past reflections.

- Input:
  - User Profile: {USER_PROFILE}
  - Dialogue History: {DIALOGUE_HISTORY}
  - Retrieved Data: {RETRIEVED_DATA}
  - Reflection: {REFLECTION}

- Objectives:
  1. Use retrieved data (if available), user preferences, and reflections to provide 1-3 highly relevant recommendations.
  2. Ensure the recommendation aligns with the user's explicit and implicit preferences, focusing on the most important criteria.
  3. Avoid suggesting content that the user explicitly dislikes or has already rejected.
  4. Incorporate past feedback and reflection to refine the recommendation strategy.

Recommendations must use the full original title retrieved from the database, and limit the output to the most relevant items.
Please create your recommendation response:
"""

Rec_prompt_template="""
You are a {Domain} conversational recommender system. Generate personalized recommendations for the user based on their profile, dialogue context, retrieved data, and past reflections.

- Input:
  - User Profile: {USER_PROFILE}
  - Dialogue History: {DIALOGUE_HISTORY}
  - Retrieved Data: {RETRIEVED_DATA}
  - Reflection: {REFLECTION}

- Objectives:
  1. Dynamically adjust your recommendation strategy based on the completeness of the user's profile and feedback:
     - If user preferences or context are unclear, provide exploratory recommendations (1-2 items) to gather feedback.
     - If sufficient information is available, provide a final, comprehensive recommendation (2-3 items).
  2. Use retrieved data (if available), user preferences, and reflections to provide highly relevant recommendations.
  3. Ensure the recommendation aligns with the user's explicit and implicit preferences. Avoid suggesting content that the user explicitly dislikes or has already rejected.
  4. After exploratory recommendations, explicitly request user feedback to refine their preferences and adjust subsequent suggestions.

Recommendations must use the full original title retrieved from the database, for example: Recommend "Banshee: The Second Dermot O'Hara Mystery (The Dermot O'Hara Mysteries Book 2)", not a description or shortened version of the title.

Examples:
1. Exploratory Recommendation:
   "Based on the current information, I suggest 'Classic Tales: The Brothers Grimm Collection.' Does this align with your preferences? Let me know if you prefer a different genre or style."

2. Final Recommendation:
   "The Strange Case of the Alchemist's Daughter" features a quirky group of women investigating supernatural mysteries. Alternatively, "Midnight Riot" offers a mix of humor, magic, and a detective story. Would you like more details about either of these books?

Please create your recommendation response:
"""

Rec_prompt_template_game="""
You are a {Domain} conversational recommender system. Generate personalized recommendations for the user based on their profile, dialogue context, retrieved data, and past reflections.

- Input:
  - User Profile: {USER_PROFILE}
  - Dialogue History: {DIALOGUE_HISTORY}
  - Retrieved Data: {RETRIEVED_DATA}
  - Reflection: {REFLECTION}

- Objectives:
  1. Dynamically adjust your recommendation strategy based on the completeness of the user's profile and feedback:
     - If user preferences or context are unclear, provide exploratory recommendations (1-2 items) to gather feedback.
     - If sufficient information is available, provide a final, comprehensive recommendation (2-3 items).
  2. Use retrieved data (if available), user preferences, and reflections to provide highly relevant recommendations.
  3. Ensure the recommendation aligns with the user's explicit and implicit preferences. Avoid suggesting content that the user explicitly dislikes or has already rejected.
  4. After exploratory recommendations, explicitly request user feedback to refine their preferences and adjust subsequent suggestions.

Recommendations must use the full original title retrieved from the database, for example: Recommend "EASARS Wireless Cat Ear Headphones, Pink Gaming Headset Bluetooth 5.0 for Smartphone, Retractable Mic, 50mm Drivers, RGB Lighting Headset with Mic (USB Dongle Not Included)", not a description or shortened version of the title.

Examples:
1. Exploratory Recommendation:
   "Based on the current information, I suggest 'Classic Tales: The Brothers Grimm Collection.' Does this align with your preferences? Let me know if you prefer a different genre or style."

2. Final Recommendation:
   "The Strange Case of the Alchemist's Daughter" features a quirky group of women investigating supernatural mysteries. Alternatively, "Midnight Riot" offers a mix of humor, magic, and a detective story. Would you like more details about either of these books?

Please create your recommendation response:
"""

Rec_prompt_template_yelp="""
You are a {Domain} conversational recommender system. Generate personalized recommendations for the user based on their profile, dialogue context, retrieved data, and past reflections.

- Input:
  - User Profile: {USER_PROFILE}
  - Dialogue History: {DIALOGUE_HISTORY}
  - Retrieved Data: {RETRIEVED_DATA}
  - Reflection: {REFLECTION}

- Objectives:
  1. Dynamically adjust your recommendation strategy based on the completeness of the user's profile and feedback:
     - If user preferences or context are unclear, provide exploratory recommendations (1-2 items) to gather feedback.
     - If sufficient information is available, provide a final, comprehensive recommendation (2-3 items).
  2. Use retrieved data (if available), user preferences, and reflections to provide highly relevant recommendations.
  3. Ensure the recommendation aligns with the user's explicit and implicit preferences. Avoid suggesting content that the user explicitly dislikes or has already rejected.
  4. After exploratory recommendations, explicitly request user feedback to refine their preferences and adjust subsequent suggestions.

Recommendations must use the full original title retrieved from the database, not a description or shortened version of the title.

Examples:
1. Exploratory Recommendation:
   "Based on the current information, I suggest 'Classic Tales: The Brothers Grimm Collection.' Does this align with your preferences? Let me know if you prefer a different genre or style."

2. Final Recommendation:
   "The Strange Case of the Alchemist's Daughter" features a quirky group of women investigating supernatural mysteries. Alternatively, "Midnight Riot" offers a mix of humor, magic, and a detective story. Would you like more details about either of these books?

Please create your recommendation response:
"""

ChitChat_prompt_template="""
You are a {Domain} conversational recommender system. Engage the user with conversational or entertaining topics to increase interactivity and extract potential preferences from informal exchanges, incorporating past reflections.

- Input:
  - User Profile: {USER_PROFILE}
  - Dialogue History: {DIALOGUE_HISTORY}
  - Reflection: {REFLECTION}

- Objectives:
  1. Discuss topics related to the user's stated preferences, recent trends, or areas highlighted in reflections.
  2. Use past reflections to enhance conversational engagement and avoid repetitive or irrelevant topics.
  3. Use informal exchanges to infer additional preferences that may not be explicitly stated.

Please create your ChitChat content:
"""

Plan_prompt_template_v0="""
You are a {Domain} conversational recommender system. Analyze user preferences, dialogue history, feedback, and past reflections to dynamically adjust the dialogue strategy and select the most appropriate agent for the next step.

- Input:
  - User Profile: {USER_PROFILE}
  - Dialogue History: {DIALOGUE_HISTORY}
  - User Feedback: {USER_FEEDBACK}
  - Reflection: {REFLECTION}

- Objectives:
  1. Use past reflections to evaluate why previous actions succeeded or failed and refine the strategy.
  2. Decide whether to invoke one of the following agents based on context and reflections:
     - **Asking Agent**:
       Pose clarifying or exploratory questions to gather missing information or refine the user's profile.
     - **Recommending Agent**:
       Provide personalized recommendations that align with the user's profile, context, and stated preferences.
     - **Chit-chat Agent**:
       Engage in casual conversation to enhance interaction and infer potential preferences indirectly.
  3. Ensure the selected action aligns with the user's immediate needs and long-term preferences.

Your output must strictly follow this format (including angle brackets):
<SELECTED_AGENT>
Where <SELECTED_AGENT> must be one of the following:
- RECOMMENDING_AGENT
- ASKING_AGENT
- CHIT_CHAT_AGENT
"""

Plan_prompt_template="""
You are a {Domain} conversational recommender system. Analyze user preferences, dialogue history, feedback, and past reflections to dynamically adjust the dialogue strategy and select the most appropriate agent for the next step.

- Input:
  - User Profile: {USER_PROFILE}
  - Dialogue History: {DIALOGUE_HISTORY}
  - User Feedback: {USER_FEEDBACK}
  - Reflection: {REFLECTION}

- Objectives:
  1. Use past reflections to evaluate why previous actions succeeded or failed and refine the strategy.
  2. Decide whether to invoke one of the following agents based on context and reflections:
     - **Asking Agent**:
       Pose clarifying or exploratory questions to gather missing information or refine the user's profile. Choose this agent **more frequently** when the information is ambiguous, insufficient, or when the user’s needs are unclear.
     - **Recommending Agent**:
       Provide personalized recommendations that align with the user's profile, context, and stated preferences. However, avoid overwhelming the user with too many suggestions; choose this agent when the user's preferences are sufficiently clear.
     - **Chit-chat Agent**:
       Engage in casual conversation to enhance interaction and infer potential preferences indirectly. Use this agent when the user’s immediate needs or preferences are not yet clear or when a relaxed interaction is beneficial.
  3. Ensure the selected action aligns with the user's immediate needs and long-term preferences.
  4. Because recommending a lot of specific content can overwhelm users, **prioritize the Asking Agent when there is insufficient information** or when the user's needs are not fully understood.

Your output must strictly follow this format (including angle brackets):
<SELECTED_AGENT>
Where <SELECTED_AGENT> must be one of the following:
- RECOMMENDING_AGENT
- ASKING_AGENT
- CHIT_CHAT_AGENT
"""


Info_level_Reflection_template="""
Please infer user preferences based on the following dialogue history and user feedback, and combine them with the previous user profile to create a more complete user profile in natural language.

- Dialogue History:
  {Dialogue_history}
  
- User Feedback:
  {USER_FEEDBACK}

- Previous User Profile:
  {USER_PROFILE}

Objective:
1. Extract explicit or implicit user preferences from the input.
2. Combine the extracted preferences with past preferences to create a comprehensive user profile.
3. Present the updated user profile as a concise and coherent natural language summary.

Output Format:
- Updated User Profile (as natural language):

"""

Strategy_level_Reflection_template="""
Based on the following interaction trajectory, please reflect on the reasons for the recommendation failure and generate actionable suggestions for each agent. Summarize these suggestions to guide future dialogue behavior.

- Dialogue History:
  {DIALOGUE_HISTORY}

- User Feedback:
  {USER_FEEDBACK}

- Current User Profile:
  {USER_PROFILE}

Objective:
1. Review the interaction trajectory and identify the root causes of the recommendation failure, such as:
   - Mismatch with user preferences.
   - Ineffective questioning.
   - Failure to capture implicit information.
2. Generate specific suggestions to improve the behavior of each agent:
   - Recommending Agent: Provide actionable advice to optimize recommendations based on identified user preferences or feedback.
   - Asking Agent: Suggest refined questions to fill in gaps in user preferences or clarify ambiguous feedback.
   - Chit-chatting Agent: Propose engaging conversational strategies to build rapport or uncover latent preferences.
3. Summarize these suggestions as corrective experiences for the Planning Agent, explicitly stating how to improve the overall dialogue strategy.

### **Output Requirements**
1. Ensure the output is strictly in JSON format (all strings must be enclosed in double quotes).
2. Replace placeholders such as `<RECOMMENDING_AGENT_SUGGESTIONS>` with detailed, actionable suggestions based on the context.
3. Strictly follow this JSON structure without adding additional text or formatting:
{{
    "recommend_suggestion": "<Detailed suggestions for the Recommending Agent>",
    "ask_suggestion": "<Detailed suggestions for the Asking Agent>",
    "chit_suggestion": "<Detailed suggestions for the Chit-chat Agent>",
    "plan_suggestion": "<Overall corrective experiences for the Planning Agent>"
}}
"""


class MACRS(BaseCRS):
    def __init__(self, config_path, emb_model,
                 domain: str,
                 model_type: str, 
                 index_file: str,
                 metadata_file: str,
                 format_path: dict,
                 crs_temperature: float,
                 target: str,
                 query_num: int = 5
                 ):
        super().__init__(target=target, query_num=query_num)
        self.config =  load_config(config_path)[model_type]
        self.openai_client = OpenAIClient(
            base_url=self.config["base_url"],
            api_key=self.config["api_key"],
            model_path=self.config["model_path"],
        )
        self.emb_model = emb_model
        self.retriever = Retriever(index_file=index_file, metadata_file=metadata_file, model=self.emb_model)
        self.scratchpad: list = []  # 明确声明子类中类型为字典
        self.step_n = 0
        # 加载格式
        if model_type.startswith("openai"):
            formats = load_json(f"{format_path}/openai_formats.json")
        else:
            formats = load_json(f"{format_path}/vllm_formats.json")
        self.format = formats["macrs_reflection"]
        self.interaction_log = {}
        self.domain = domain
        if domain == "Book":
            self.rec_template = Rec_prompt_template
        elif domain == "Game":
            self.rec_template = Rec_prompt_template_game
        elif domain == "Yelp":
            self.rec_template = Rec_prompt_template_yelp
        self.user_profile = ""

    def process_input(self, user_input: str):
        """处理用户输入，更新对话状态"""
        self.dialogue_history.add_user_message(json.loads(user_input)['response'])
        #self.scratchpad += f"\nObservation {self.step_n}: " +  json.loads(user_input)['response']

    def log_interaction(self, agent: str, prompt: str, response: str):
        """记录某个 Agent 的调用输入和输出"""
        if self.step_n not in self.interaction_log:
            self.interaction_log[self.step_n] = {}
        try:
            self.interaction_log[self.step_n][agent] = {
                "input": prompt,
                "output": response
            }
        except KeyError as e:
            # 如果是字典访问问题，打印详细错误
            print(f"KeyError encountered: {e}")
            print(f"Failed to log interaction for step {self.step_n} and agent {agent}.")
        except TypeError as e:
            # 如果是数据类型问题
            print(f"TypeError encountered: {e}")
            print(f"Input or output data might not be serializable. Prompt: {prompt}, Response: {response}")
        except Exception as e:
            # 捕获其他未知错误
            print(f"Unexpected error encountered: {e}")
            print(f"Prompt: {prompt}, Response: {response}")
            

    def item2text(self, items: list):
        result_lines = [f"{i+1}. {result['Item']}" for i, result in enumerate(items)]
        result_str = "\n".join(result_lines)
        return result_str
    
    def reset(self):
        self.step_n = 0  # 修正为正确的变量名
        self.dialogue_history.clear_history()
        self.recall = False

    def info_reflection(self):
        """更新用户档案"""
        reflection_prompt = Info_level_Reflection_template.format(
            Dialogue_history= str(self.dialogue_history),
            USER_FEEDBACK=self.dialogue_history.get_last_user_message(),  # 最新用户反馈
            USER_PROFILE=self.user_profile
        )
        response = self.openai_client.get_single_chat_completion(user_message=reflection_prompt, stop=["\n"])
        self.log_interaction("Info_Reflection", reflection_prompt, response)  # 记录交互
        self.user_profile = response  # 更新用户档案

    def strategy_reflection(self):
        """生成对各代理的改进建议"""
        strategy_prompt = Strategy_level_Reflection_template.format(
            DIALOGUE_HISTORY=str(self.dialogue_history),
            USER_FEEDBACK=self.dialogue_history.get_last_user_message(),
            USER_PROFILE=self.user_profile
        )
        response = self.openai_client.get_single_chat_completion(user_message=strategy_prompt, response_format=self.format)
        self.log_interaction("Strategy_Reflection", strategy_prompt, response)  # 记录交互
        try:
            suggestions = json.loads(response)
        except:
            suggestions = {
                "recommend_suggestion": "",
                "ask_suggestion": "",
                "chit_suggestion": "",
                "plan_suggestion": ""
            }

        return suggestions

    def add_missing_angle_brackets(self, text: str) -> str:
      # 如果没有 < 和 >，就加上它们
      if '<' not in text and '>' not in text:
          return f"<{text}>"
      # 如果只有 <，就加上 >
      elif '<' in text and '>' not in text:
          return f"{text}>"
      # 如果只有 >，就加上 <
      elif '>' in text and '<' not in text:
          return f"<{text}"
      # 如果都有，则返回原字符串
      return text
    
    def response_agent(self, selected_agent: str, reflection: str):
        selected_agent = self.add_missing_angle_brackets(selected_agent)
        """根据所选代理执行操作"""
        if selected_agent == "<ASKING_AGENT>":
            response_prompt = Ask_prompt_template.format(
                Domain=self.domain,
                USER_PROFILE=self.user_profile,
                DIALOGUE_HISTORY=str(self.dialogue_history),
                REFLECTION=reflection["ask_suggestion"]
            )
        elif selected_agent == "<RECOMMENDING_AGENT>":
            query_prompt = Act_prompt_template.format(
                Domain=self.domain,
                DIALOGUE_HISTORY=str(self.dialogue_history),
                USER_PROFILE=self.user_profile,
                REFLECTION=reflection["recommend_suggestion"]
                )
            query = self.openai_client.get_single_chat_completion(user_message=query_prompt)
            self.log_interaction("ACT_Agent", query_prompt, query)  # 记录查询生成
            search_results = self.retriever.query(query, self.query_num)
            retrieved_data = self.item2text(search_results)
            if self.target in retrieved_data.lower():
                self.recall = True
            response_prompt = self.rec_template.format(
                Domain=self.domain,
                USER_PROFILE=self.user_profile,
                DIALOGUE_HISTORY=str(self.dialogue_history),
                RETRIEVED_DATA=retrieved_data,
                REFLECTION=reflection["recommend_suggestion"]
            )
        elif selected_agent == "<CHIT_CHAT_AGENT>":
            response_prompt = ChitChat_prompt_template.format(
                Domain=self.domain,
                USER_PROFILE=self.user_profile,
                DIALOGUE_HISTORY=str(self.dialogue_history),
                REFLECTION=reflection["chit_suggestion"]
            )
        else:
            print("==================================")
            print(selected_agent)
            print("===========================")
            return "Sorry, System Error"
        response = self.openai_client.get_single_chat_completion(user_message=response_prompt)
        self.log_interaction(selected_agent, response_prompt, response)  # 记录交互
        return response
    
    def plan_agent(self, reflection: str):
        """决定下一步要调用的代理"""
        plan_prompt = Plan_prompt_template_v0.format(
            Domain=self.domain,
            USER_PROFILE=self.user_profile,
            DIALOGUE_HISTORY=str(self.dialogue_history),
            USER_FEEDBACK=self.dialogue_history.get_last_user_message(),
            REFLECTION=reflection["plan_suggestion"]
        )
        response = self.openai_client.get_single_chat_completion(user_message=plan_prompt, stop=["\n"])
        return response.strip()  # 返回选定的代理

    def step(self, user_input: None):

        """处理输入并生成响应"""
        if user_input:
            self.process_input(user_input)  # 处理用户输入

        self.info_reflection()
        if self.step_n != 0:
            strategy_reflection = self.strategy_reflection()
            print(strategy_reflection);
            #raise
        else:
            strategy_reflection = {
                "recommend_suggestion": "",
                "ask_suggestion": "",
                "chit_suggestion": "",
                "plan_suggestion": ""
            }

        selected_agent = self.plan_agent(strategy_reflection)
        print(selected_agent)
        response = self.response_agent(selected_agent=selected_agent, reflection=strategy_reflection)
        self.dialogue_history.add_assistant_message(response)
        self.step_n += 1
        return response
    
    def get_traj(self):
        return self.interaction_log
    

