from .base_crs import BaseCRS
from model.model import OpenAIClient
from utils import load_config, load_json, load_jsonl
from .tools.retriever import Retriever
import json
import re
from sentence_transformers import SentenceTransformer

User_prompt_template_fs_book="""Solve recommendation tasks by interleaving "Observation", "Thought", and "Action" steps.

Thought: Analyze the user's request, reason about the current situation, and strategize the next step to uncover user preferences or provide recommendations.
Action: Choose one of the following actions based on the situation:
(1) Search[Keywords]: Search for relevant information using targeted keywords to aid in generating specific suggestions. After completing the Thought step, output "Search[Keywords]," and results will be returned in the "Observation" step.

(2) Ask[Question]: Interact with the user to clarify their preferences or constraints. Use strategic questioning to gradually uncover their requirements. After completing the Thought step, output "Ask[Question]" and wait for the user's response.

(3) Recommend[Answer]: Provide recommendations based on available information and reasoning. Recommendations can serve two purposes:

Exploratory Purpose: Provide a tentative recommendation when information is incomplete to gather user feedback and refine preferences.
Final Purpose: Provide a clear and comprehensive recommendation when sufficient information is available.

Recommendations must use the full original title retrieved from the database, for example: Recommend "Banshee: The Second Dermot O'Hara Mystery (The Dermot O'Hara Mysteries Book 2)", not a description or shortened version of the title.

(4) Response[Content]: Address user inquiries, handle off-topic remarks, or respond to unrelated requests to maintain a natural and coherent conversation flow.

You may take as many steps as necessary to guide the user toward the most suitable recommendation. Below is an example of how to approach the task: {examples} (END OF EXAMPLES) Please input the thought or action directly, do not input the additional "Action n" or "Thought n". Dialogue:\n{scratchpad}"""



User_prompt_template_zs_book="""Solve recommendation tasks by interleaving "Observation", "Thought", and "Action" steps.

Thought: Analyze the user's request, reason about the current situation, and strategize the next step to uncover user preferences or provide recommendations.
Action: Choose one of the following actions based on the situation:
(1) Search[Keywords]: Search for relevant information using targeted keywords to aid in generating specific suggestions. After completing the Thought step, output "Search[Keywords]," and results will be returned in the "Observation" step.

(2) Ask[Question]: Interact with the user to clarify their preferences or constraints. Use strategic questioning to gradually uncover their requirements. After completing the Thought step, output "Ask[Question]" and wait for the user's response.

(3) Recommend[Answer]: Provide recommendations based on available information and reasoning. Recommendations can serve two purposes:

Exploratory Purpose: Provide a tentative recommendation when information is incomplete to gather user feedback and refine preferences.
Final Purpose: Provide a clear and comprehensive recommendation when sufficient information is available.

Recommendations must use the full original title retrieved from the database, for example: Recommend "Banshee: The Second Dermot O'Hara Mystery (The Dermot O'Hara Mysteries Book 2)", not a description or shortened version of the title.

(4) Response[Content]: Address user inquiries, handle off-topic remarks, or respond to unrelated requests to maintain a natural and coherent conversation flow.

You may take as many steps as necessary to guide the user toward the most suitable recommendation. Please input the thought or action directly, do not input the additional "Action n" or "Thought n". Dialogue:\n{scratchpad}"""

User_prompt_template_fs_game="""Solve recommendation tasks by interleaving "Observation", "Thought", and "Action" steps.

Thought: Analyze the user's request, reason about the current situation, and strategize the next step to uncover user preferences or provide recommendations.
Action: Choose one of the following actions based on the situation:
(1) Search[Keywords]: Search for relevant information using targeted keywords to aid in generating specific suggestions. After completing the Thought step, output "Search[Keywords]," and results will be returned in the "Observation" step.

(2) Ask[Question]: Interact with the user to clarify their preferences or constraints. Use strategic questioning to gradually uncover their requirements. After completing the Thought step, output "Ask[Question]" and wait for the user's response.

(3) Recommend[Answer]: Provide recommendations based on available information and reasoning. Recommendations can serve two purposes:

Exploratory Purpose: Provide a tentative recommendation when information is incomplete to gather user feedback and refine preferences.
Final Purpose: Provide a clear and comprehensive recommendation when sufficient information is available.

Recommendations must use the full original title retrieved from the database, for example: Recommend "EASARS Wireless Cat Ear Headphones, Pink Gaming Headset Bluetooth 5.0 for Smartphone, Retractable Mic, 50mm Drivers, RGB Lighting Headset with Mic (USB Dongle Not Included)", not a description or shortened version of the title.

(4) Response[Content]: Address user inquiries, handle off-topic remarks, or respond to unrelated requests to maintain a natural and coherent conversation flow.

You may take as many steps as necessary to guide the user toward the most suitable recommendation. Below is an example of how to approach the task: {examples} (END OF EXAMPLES) Please input the thought or action directly, do not input the additional "Action n" or "Thought n". Dialogue:\n{scratchpad}"""


User_prompt_template_zs_game="""Solve recommendation tasks by interleaving "Observation", "Thought", and "Action" steps.

Thought: Analyze the user's request, reason about the current situation, and strategize the next step to uncover user preferences or provide recommendations.
Action: Choose one of the following actions based on the situation:
(1) Search[Keywords]: Search for relevant information using targeted keywords to aid in generating specific suggestions. After completing the Thought step, output "Search[Keywords]," and results will be returned in the "Observation" step.

(2) Ask[Question]: Interact with the user to clarify their preferences or constraints. Use strategic questioning to gradually uncover their requirements. After completing the Thought step, output "Ask[Question]" and wait for the user's response.

(3) Recommend[Answer]: Provide recommendations based on available information and reasoning. Recommendations can serve two purposes:

Exploratory Purpose: Provide a tentative recommendation when information is incomplete to gather user feedback and refine preferences.
Final Purpose: Provide a clear and comprehensive recommendation when sufficient information is available.

Recommendations must use the full original title retrieved from the database, for example: Recommend "EASARS Wireless Cat Ear Headphones, Pink Gaming Headset Bluetooth 5.0 for Smartphone, Retractable Mic, 50mm Drivers, RGB Lighting Headset with Mic (USB Dongle Not Included)", not a description or shortened version of the title.

(4) Response[Content]: Address user inquiries, handle off-topic remarks, or respond to unrelated requests to maintain a natural and coherent conversation flow.

You may take as many steps as necessary to guide the user toward the most suitable recommendation. Please input the thought or action directly, do not input the additional "Action n" or "Thought n". Dialogue:\n{scratchpad}"""

User_prompt_template_fs_yelp="""Solve recommendation tasks by interleaving "Observation", "Thought", and "Action" steps.

Thought: Analyze the user's request, reason about the current situation, and strategize the next step to uncover user preferences or provide recommendations.
Action: Choose one of the following actions based on the situation:
(1) Search[Keywords]: Search for relevant information using targeted keywords to aid in generating specific suggestions. After completing the Thought step, output "Search[Keywords]," and results will be returned in the "Observation" step.

(2) Ask[Question]: Interact with the user to clarify their preferences or constraints. Use strategic questioning to gradually uncover their requirements. After completing the Thought step, output "Ask[Question]" and wait for the user's response.

(3) Recommend[Answer]: Provide recommendations based on available information and reasoning. Recommendations can serve two purposes:

Exploratory Purpose: Provide a tentative recommendation when information is incomplete to gather user feedback and refine preferences.
Final Purpose: Provide a clear and comprehensive recommendation when sufficient information is available.

Recommendations must use the full original title retrieved from the database, not a description or shortened version of the title.

(4) Response[Content]: Address user inquiries, handle off-topic remarks, or respond to unrelated requests to maintain a natural and coherent conversation flow.

You may take as many steps as necessary to guide the user toward the most suitable recommendation. Below is an example of how to approach the task: {examples} (END OF EXAMPLES) Please input the thought or action directly, do not input the additional "Action n" or "Thought n". Dialogue:\n{scratchpad}"""


User_prompt_template_zs_yelp="""Solve recommendation tasks by interleaving "Observation", "Thought", and "Action" steps.

Thought: Analyze the user's request, reason about the current situation, and strategize the next step to uncover user preferences or provide recommendations.
Action: Choose one of the following actions based on the situation:
(1) Search[Keywords]: Search for relevant information using targeted keywords to aid in generating specific suggestions. After completing the Thought step, output "Search[Keywords]," and results will be returned in the "Observation" step.

(2) Ask[Question]: Interact with the user to clarify their preferences or constraints. Use strategic questioning to gradually uncover their requirements. After completing the Thought step, output "Ask[Question]" and wait for the user's response.

(3) Recommend[Answer]: Provide recommendations based on available information and reasoning. Recommendations can serve two purposes:

Exploratory Purpose: Provide a tentative recommendation when information is incomplete to gather user feedback and refine preferences.
Final Purpose: Provide a clear and comprehensive recommendation when sufficient information is available.

Recommendations must use the full original title retrieved from the database, not a description or shortened version of the title.

(4) Response[Content]: Address user inquiries, handle off-topic remarks, or respond to unrelated requests to maintain a natural and coherent conversation flow.

You may take as many steps as necessary to guide the user toward the most suitable recommendation. Please input the thought or action directly, do not input the additional "Action n" or "Thought n". Dialogue:\n{scratchpad}"""

few_shot_examples_book="""
Observation 0: I would like to read a mysterious and thrilling book, any suggestions?
Thought 0: The user has expressed interest in finding a mysterious and thrilling book. To narrow down the recommendations, I should clarify their preferences for subgenres or specific themes in mysteries.
Action 0: Ask[Do you prefer mysteries with supernatural elements, psychological suspense, or crime-focused themes?]
Observation 1: I love mysteries with a supernatural twist, especially if there are quirky characters involved.
Thought 1: The user enjoys supernatural mysteries with quirky characters. Based on this preference, I should look for books that include humor, supernatural elements, and intriguing characters.
Action 1: Search[Supernatural mystery books with quirky characters]
Observation 2: Results found:
1. Title: The Strange Case of the Alchemist's Daughter;  Categories: ... Description: ...
2. Title: Midnight Riot" (Rivers of London series);  Categories: ... Description: ...
3. Title: The House with a Clock in Its Walls;  Categories: ... Description: ...
Thought 2: These results align with the user’s preferences for supernatural mysteries. To ensure the recommendation is tailored, I will suggest two books from the list and ask for feedback.
Action 2: Recommend["The Strange Case of the Alchemist's Daughter" features a quirky group of women investigating supernatural mysteries. Alternatively, "Midnight Riot" offers a mix of humor, magic, and a detective story. Would you like more details about either of these books?]
Observation 3: The Strange Case of the Alchemist's Daughter sounds great! Tell me more about it.
Thought 3: The user is interested in "The Strange Case of the Alchemist's Daughter." I will provide additional details to help them make an informed decision.
Action 3: Response["The Strange Case of the Alchemist's Daughter" is set in Victorian London and follows Mary Jekyll as she discovers a secret society tied to her father's mysterious past. Along with quirky companions like Sherlock Holmes and Dr. Watson, Mary uncovers supernatural secrets. It's a blend of humor, mystery, and magic."]
"""

few_shot_examples_game="""
Observation 0: I am looking for a fun single-player game with an immersive experience. Any recommendations? 
Thought 0: The user has expressed interest in finding an immersive single-player game. To provide relevant recommendations, I should clarify whether they prefer story-driven experiences or open-world exploration.
Action 0: Ask[Do you prefer story-driven games with complex characters, or are you more interested in open-world exploration?]
Observation 1: I prefer story-driven games, especially those with complex characters and interesting choice systems. 
Thought 1: The user enjoys story-driven games with a focus on complex characters and choice systems. Based on this preference, I should search for games that emphasize narrative depth, meaningful decisions, and character development.
Action 1: Search[story-driven games with complex characters and choice systems]
Observation 2: Results found: 
1. Title: Detroit: Become Human; Categories: Adventure/Interactive Movie; Description: Players control three androids, making choices that affect the story. 
2. Title: Life is Strange; Categories: Adventure; Description: A narrative-driven adventure where player choices impact the storyline. 
3. Title: The Witcher 3: Wild Hunt; Categories: RPG; Description: An open-world RPG with complex characters and a rich storyline.
Thought 2: The search results align well with the user's interest in story-driven games. To tailor the recommendation, I will present two options that emphasize narrative and character interaction and ask for their feedback.
Action 2: Recommend["Detroit: Become Human" is an interactive narrative game where players control android characters and make impactful choices. Alternatively, "Life is Strange" is known for its emotionally-driven narrative and deep character interactions. Would you like more details about either of these games?]
"""


few_shot_examples_yelp="""
Observation 0: I'm looking for a family-friendly restaurant for a weekend dinner. Any recommendations?
Thought 0: The user wants a family-friendly restaurant for dinner on the weekend. To provide relevant recommendations, I should ask about their preferred cuisine and any specific location preferences.
Action 0: Ask[What type of cuisine are you interested in? Do you have any preferred locations or neighborhoods?]
Observation 1: We love Italian food and would prefer something in the downtown area with a cozy atmosphere.
Thought 1: The user prefers Italian cuisine, a downtown location, and a cozy atmosphere. Based on these preferences, I should search for Italian restaurants in the downtown area that are known for their family-friendly and cozy environments.
Action 1: Search[family-friendly Italian restaurants in downtown area with cozy atmosphere]
Observation 2: Search results:
1. Name: Bella Italia; Address: 123 Main St, Downtown; Rating: 4.6; Description: Traditional Italian dishes with a warm and inviting atmosphere, perfect for families.
2. Name: Mario's Trattoria; Address: 456 Elm St, Downtown; Rating: 4.5; Description: Authentic Italian cuisine offering a cozy dining experience and a kids' menu.
3. Name: Luigi's Pizzeria; Address: 789 Oak St, Downtown; Rating: 4.4; Description: Family-friendly pizzeria with a relaxed environment and a variety of Italian favorites.
Thought 2: The search results align well with the user's preferences. To help the user decide, I will recommend the top two highest-rated restaurants and ask if they need more details.
Action 2: Recommend["Bella Italia" has a rating of 4.6 and is known for its traditional Italian dishes and warm atmosphere, making it perfect for family dinners. Alternatively, "Mario's Trattoria" is rated 4.5 and offers an authentic Italian dining experience with a dedicated kids' menu. Would you like more information about either of these restaurants?]
"""



class ReActCRS(BaseCRS):
    def __init__(self, config_path, emb_model,
                 domain: str,
                 model_type: str, 
                 index_file: str,
                 metadata_file: str,
                 shot_type: str,
                 lora: None,
                 crs_temperature: float,
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
        self.reply_count = 0
        self.emb_model = emb_model
        self.retriever = Retriever(index_file=index_file, metadata_file=metadata_file, model=self.emb_model)
        self.scratchpad = ""
        self.step_n = 0
        self.shot_type = shot_type
        self.domain = domain
        if domain == "Book":
            if shot_type == "few_shot":
                self.crs_template = User_prompt_template_fs_book
                self.examples = few_shot_examples_book
            elif shot_type == "zero_shot":
                self.crs_template = User_prompt_template_zs_book
        elif domain == "Game":
            if shot_type == "few_shot":
                self.crs_template = User_prompt_template_fs_game
                self.examples = few_shot_examples_game
            elif shot_type == "zero_shot":
                self.crs_template = User_prompt_template_zs_game
        elif domain == "Yelp":
            if shot_type == "few_shot":
                self.crs_template = User_prompt_template_fs_yelp
                self.examples = few_shot_examples_yelp
            elif shot_type == "zero_shot":
                self.crs_template = User_prompt_template_zs_yelp
        self.interaction_log = {}

    def process_input(self, user_input: str):
        """处理用户输入，更新对话状态"""
        self.dialogue_history.add_user_message(json.loads(user_input)['response'])
        self.scratchpad += f"\nObservation {self.step_n}: " +  json.loads(user_input)['response']

    def item2text(self, items: list):
        result_lines = [f"{i+1}. {result['Item']}" for i, result in enumerate(items)]
        result_str = "\n".join(result_lines)
        return result_str
    
    def reset(self):
        self.step_n = 0  # 修正为正确的变量名
        self.dialogue_history.clear_history()
        self.recall = False
        self.reply_count = 0

    def log_interaction(self, action: str, prompt: str, response: str):
        #error的数据跳过记录
        if self.reply_count not in self.interaction_log:
            self.interaction_log[self.reply_count] = {"reward": 0}  # 默认 binary 为 1
        
        step_data = self.interaction_log[self.reply_count]
        try:
            if action == "thought":
                step_data[f"thought {self.step_n}"] = {
                    "input": prompt,
                    "output": response
                }
            elif action == "search":
                step_data[f"action {self.step_n}"] = {
                    "input": prompt,
                    "output": response
                }
            else:
                step_data[f"action {self.step_n}"] = {
                    "input": prompt,
                    "output": response
                }
                self.reply_count += 1
        except KeyError as e:
            print(f"KeyError: {e} - Step: {self.step_n}, Reply count: {self.reply_count}")
            raise
        except TypeError as e:
            print(f"TypeError: {e} - Invalid data encountered for step {self.step_n}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    def step(self, user_input: None):

        """处理输入并生成响应"""
        if user_input:
            self.process_input(user_input)  # 处理用户输入
        #Format成obs
        self.scratchpad += f'\nThought {self.step_n}:'
        if self.shot_type == "few_shot":
            user_message = self.crs_template.format(Domain=self.domain, examples=self.examples, scratchpad=self.scratchpad)
        else:
            user_message = self.crs_template.format(Domain=self.domain, scratchpad=self.scratchpad)
        Thought = ' ' + self.openai_client.get_single_chat_completion(user_message=user_message, stop=["\n"])
        self.log_interaction("thought", user_message, Thought)
        self.scratchpad += ' ' + Thought
        self.scratchpad += f'\nAction {self.step_n}:'

        #print(self.scratchpad)
        if self.shot_type == "few_shot":
            user_message = self.crs_template.format(Domain=self.domain, examples=self.examples, scratchpad=self.scratchpad)
        else:
            user_message = self.crs_template.format(Domain=self.domain, scratchpad=self.scratchpad)
        #print(user_message);
        #raise

        for i in range(3):
            try:
                action = self.openai_client.get_single_chat_completion(user_message=user_message, stop=["]"]) + "]"
            except:
                continue
        
        
        self.scratchpad += ' ' + action
        #self.step_n += 1
        print(action)
        # 解析 Action
        parsed_action = self.parse_action(action)
        response = "Sorry, System Error" #默认错误回复，如果生成了正确回复就会更新。
        #Obs
        #self.scratchpad += f'\nObservation {self.step_n}:'
        if parsed_action:
            action_type, argument = parsed_action
            action_type = action_type.lower()

            if action_type == "search":
                self.log_interaction("search", user_message, action)
                self.step_n += 1
                # 执行搜索操作并生成 Observation
                search_results = self.retriever.query(argument, self.query_num)
                observation = self.item2text(search_results)
                if self.target in observation.lower():
                    self.recall = True
                self.scratchpad +=  f"\nObservation {self.step_n}: " +  observation

                # 生成新的 Thought 和 Action
                for i in range(3):  # 支持连续搜索的循环, 最多检索三次
                    self.scratchpad += f"\nThought {self.step_n}:"
                    if self.shot_type == "few_shot":
                        next_prompt = self.crs_template.format(Domain=self.domain, examples=self.examples, scratchpad=self.scratchpad)
                    else:
                        next_prompt = self.crs_template.format(Domain=self.domain, scratchpad=self.scratchpad)
                    new_thought = self.openai_client.get_single_chat_completion(user_message=next_prompt, stop=["\n"])
                    self.log_interaction("thought", next_prompt, new_thought)
                    self.scratchpad += f" {new_thought}"

                    self.scratchpad += f"\nAction {self.step_n}:"
                    if self.shot_type == "few_shot":
                        next_prompt = self.crs_template.format(Domain=self.domain, examples=self.examples, scratchpad=self.scratchpad)
                    else:
                        next_prompt = self.crs_template.format(Domain=self.domain, scratchpad=self.scratchpad)
                    for i in range(3):
                        try:
                            new_action = self.openai_client.get_single_chat_completion(user_message=next_prompt, stop=["]"]) + "]"
                        except:
                            continue
                    self.scratchpad += f" {new_action}"

                    # 解析新的 Action
                    print(new_action)
                    new_parsed_action = self.parse_action(new_action)
                    
                    if not new_parsed_action:
                        self.scratchpad += "Invalid Action. Valid Actions are Search[<info>], Ask[<question>], Recommend[<item>] and Response[<response>]."
                        response = "Sorry, System Error"
                        break

                    new_action_type, new_argument = new_parsed_action
                    if new_action_type == "search":
                        # 执行新的搜索操作
                        self.log_interaction("search", next_prompt, new_action)

                        search_results = self.retriever.query(new_argument, self.query_num)
                        observation = self.item2text(search_results)
                        if self.target in observation.lower():
                            self.recall = True
                        self.scratchpad += observation
                        self.step_n += 1  # 增加步骤计数器，进入下一轮
                    elif new_action_type in ["response", "ask", "recommend"]:
                        # 如果不是搜索，则生成最终回复并结束循
                        self.log_interaction(new_action_type, next_prompt, new_action)
                        response = new_argument
                        self.step_n += 1
                        break
                    else:
                        # 无效的 Action
                        self.scratchpad += "Invalid Action. Valid Actions are Search[<info>], Ask[<question>], Recommend[<item>] and Response[<response>]."
                        self.step_n += 1
                        response = "Sorry, System Error"
                        break
            elif action_type in ["response", "ask", "recommend"]:
                # 对于直接的回复、提问或推荐，直接使用 Argument 作为 Observation
                #self.scratchpad += argument
                self.log_interaction(action_type, user_message, action)
                self.step_n += 1
                response = argument
            else:
                # 无效的 Action
                self.scratchpad += "Invalid Action. Valid Actions are Search[<info>], Ask[<question>], Recommend[<item>] and Response[<response>]."
                print("unexpected action")
                self.step_n += 1
                response = "Sorry, System Error"

        else:
            # 无效的 Action
            self.step_n += 1
            self.scratchpad += "Invalid Action. Valid Actions are Search[<info>], Ask[<question>], Recommend[<item>] and Response[<response>]."
            print("parse action error")
            response = "Sorry, System Error"

        self.dialogue_history.add_assistant_message(response)
        return response
    

    # def parse_action(self, string):
    #     # 支持两种格式：
    #     # 1. "Action 2: Search[circus setting mysterious books with quirky characters and supernatural elements]"
    #     # 2. "Search[circus setting mysterious books with quirky characters and supernatural elements]"
    #     pattern = r'^(?:Action\s+\d+:\s+)?(\w+)\[(.+)\]$'
        
    #     match = re.match(pattern, string)
    #     if match:
    #         action_type = match.group(1).lower()  # 提取动作类型并转换为小写
    #         argument = match.group(2)  # 提取参数，保留原始大小写
    #         return action_type, argument
    #     else:
    #         return None
        
    def parse_action(self, string: str):
        """
        解析字符串，支持以下格式：
        1) "Action 2: Recommend[...]"
        2) "Recommend[...]"
        
        * 允许外层 bracket 中有嵌套，比如 "xxx[yyy[zzzz],"
        如果发现内部 '[' 大于 ']'，会在末尾自动补足缺失的 ']'。

        最终返回 (action_type, argument)：
        - action_type 转为小写 (如 "search", "recommend")
        - argument 为补齐后最外层 '[' ']' 之间的完整内容
            （并保留嵌套 bracket，哪怕原本没配对，也会在末尾统一补上）
        
        若字符串不符合外层动作格式(如无动作类型)或无法匹配到外层 '['，
        则返回 None。
        """

        # 1) 去除前后空格
        string = string.strip()

        # 2) 可选前缀: "Action <数字>:" (例如 "Action 2:")
        #    然后是一个单词(\w+)动作类型(Recommend, Search等)。
        prefix_pattern = r'^(?:Action\s+\d+:\s+)?(\w+)'
        match_prefix = re.match(prefix_pattern, string)
        if not match_prefix:
            return None  # 找不到动作类型

        # 提取到的动作类型，小写化
        action_type = match_prefix.group(1).lower()

        # 3) 在前缀之后，找到第一个 '[' 的位置，视为外层 bracket 的起点
        start_search_idx = match_prefix.end()
        bracket_start = string.find('[', start_search_idx)
        if bracket_start == -1:
            # 未找到任何 '[', 说明没有参数区域
            return None

        # 4) 从 bracket_start 开始，遍历字符做“栈计数”
        #    - 每遇到 '[', stack++ ；遇到 ']', stack--
        #    - 如果到达 string 末尾时 stack>0，说明缺少 ']'，自动补齐
        result_chars = []    # 用来收集从 '[' 开始到补齐后的所有字符
        bracket_count = 0
        inside_region = string[bracket_start:]  # 从第一个 '[' 到末尾的所有内容

        i = 0
        while i < len(inside_region):
            ch = inside_region[i]
            result_chars.append(ch)
            if ch == '[':
                bracket_count += 1
            elif ch == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    # 已经配对到最外层 bracket 关闭，则可以停止
                    i += 1
                    break
            i += 1
        
        # 如果离开循环时 bracket_count > 0，说明还没配对完，把缺失的 `]` 全部补上
        while bracket_count > 0:
            result_chars.append(']')
            bracket_count -= 1
        
        # 到这里，result_chars 中已经包含从第一个 '[' 开始，到最外层 ']'（含自动补齐）为止的完整内容
        # i 是我们消耗的字符数，i 可能没有到 inside_region 末尾
        bracket_section = ''.join(result_chars)

        # 5) 若最外层 bracket_section 不以 ']' 结尾，说明没有成功补到最外层，视为不合法
        if not bracket_section.endswith(']'):
            # 理论上不会出现这种情况，因为上面已经强行补足
            return None

        # 6) 提取最外层 bracket 中的内容，即去掉开头的 '[' 与结尾的 ']'
        #    strip() 可选，看你要不要保留内部首尾空格
        argument = bracket_section[1:-1].strip()

        return action_type, argument
    
    def get_state(self):
        return self.scratchpad, self.interaction_log, self.step_n, self.reply_count, self.dialogue_history
    
    def update_state(self, scratchpad, interaction_log, step_n, reply_count, dialogue_history):
        self.scratchpad = scratchpad
        self.interaction_log = interaction_log
        self.step_n = step_n
        self.reply_count = reply_count
        self.dialogue_history = dialogue_history

    def get_traj(self):
        return self.interaction_log