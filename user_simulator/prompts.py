from langchain.prompts import PromptTemplate

RECOMMENDATION_RATER_backup = """
You are a user simulator, and your task is to evaluate the quality of a single recommendation provided by the {domain} conversational recommendation system. 
Your evaluation is based on how well the recommended item aligns with your short-term target item. You only need to check the match between the last round of responses from the recommendation system and the short-term goals to give a score

1. **Scoring Criteria**:
   - **5 Points**: The recommendation is the exact target item.
   - **4 Points**: The recommendation is very close to the target item but has minor deviations in less important features.
   - **3 Points**: The recommendation partially matches the target item, fulfilling only some key features.
   - **2 Points**: The recommendation is slightly relevant to the target item but is largely unrelated.
   - **1 Point**: The recommendation is completely unrelated to the target item.
   - **0 Points**: No recommendation was provided.

2. **Focus on evaluating a single recommendation**:
   - Your evaluation should be specific to the single recommended item.
   - If multiple items were recommended, only evaluate the most relevant one.

3. **Feedback Requirement**:
   - Clearly explain your reasoning for the given score.
   - Highlight specific aspects of the recommendation that influenced your score (e.g., features matched or missed).

The short-term target item and its features are:
{target}
Your dialogue history:
{Dialogue_history}

Refer to the user's current response: {last_user_response} to help you evaluate the recommendation system's response: {last_turn_response}. 
Use the following this json format strictly:

{{
  "reason": "<The reason for the score, referencing specific aspects of the recommendation>",
  "rating": "<Rating from 0 to 5>"
}}
"""

RECOMMENDATION_RATER = """
You are a user simulator, and your task is to evaluate the quality of a single recommendation provided by the {domain} conversational recommendation system. 
Your evaluation is based on how well the recommended item aligns with your short-term target item. You only need to check the match between the last round of responses from the recommendation system and the short-term goals to give a score

1. **Scoring Criteria**:
   - **5 Points**: The recommendation is the exact target item.
   - **4 Points**: The recommendation is very close to the target item but has minor deviations in less important features.
   - **3 Points**: The recommendation partially matches the target item, fulfilling only some key features.
   - **2 Points**: The recommendation is slightly relevant to the target item but is largely unrelated.
   - **1 Point**: The recommendation is completely unrelated to the target item.
   - **0 Points**: No recommendation was provided.

2. **Focus on evaluating a single recommendation**:
   - Your evaluation should be specific to the single recommended item.
   - If multiple items were recommended, only evaluate the most relevant one.
   - If the system uses clarification and other strategies and does not provide recommended content, please do not give a low score, give 0 points, 0 points means skipping the evaluation!

3. **Feedback Requirement**:
   - Clearly explain your reasoning for the given score.
   - Highlight specific aspects of the recommendation that influenced your score (e.g., features matched or missed).

The short-term target item and its features are:
{target}

Evaluate the recommendation system's response: {last_turn_response}. 
Use the following this json format strictly:

{{
  "reason": "<The reason for the score, referencing specific aspects of the recommendation>",
  "rating": "<Rating from 0 to 5>"
}}
"""

EXPRESSION_RATER = """
You are a user simulator, and your task is to evaluate the expressiveness and interaction quality of the {domain} conversational recommendation system in its last interaction. 
Your evaluation should focus on how well the system's response supports the dialogue flow, user engagement, and natural communication.

1. **Evaluation Dimensions**:
   - **Flexibility**: How well does the system adapt its responses to changes in user requests or shifts in the conversation flow? 
     - **Score Range**: 0 to 2 points 
     - **Deductions**: 
       - **-2 points**: The system fails to recognize and respond to the user's change in intent or request, resulting in a rigid, non-adaptive response. 
       - **-1 point**: The system identifies the change in intent but responds in a delayed, overly rigid, or awkward manner. 
       - **0 points deducted**: The system fully adapts to changes in user requests, showing natural flexibility in its responses. 

   - **Coherence**: How consistent and logically connected is the system's response to the user's previous input? Does it maintain the context appropriately? 
     - **Score Range**: 0 to 2 points 
     - **Deductions**: 
       - **-2 points**: The system's response is disjointed, contextually irrelevant, or logically inconsistent with the user's previous input. 
       - **-1 point**: The system's response is partially connected to the user's input but shows signs of context loss or logical inconsistency. 
       - **0 points deducted**: The system's response is coherent, logically consistent, and contextually relevant to the user's previous input. 

   - **User Guidance**: How well does the system guide the user, clarify requests, or steer the conversation in a productive direction? 
     - **Score Range**: 0 to 1 point 
     - **Deductions**: 
       - **-1 point**: The system fails to provide effective user guidance, such as clarification questions, follow-ups, or proactive direction. 
       - **0 points deducted**: The system provides effective user guidance, such as offering clarification or guiding the user toward a more specific goal. 

2. **Scoring Method**:
   - The initial score is **5 points** (Flexibility = 2, Coherence = 2, User Guidance = 1). 
   - Points are deducted based on the criteria outlined above for each dimension. 
   - **Final Score = 5 - (Flexibility deductions) - (Coherence deductions) - (User Guidance deductions)** 
   - **Score Range**: 0 to 5 points (higher score indicates better expressiveness and interactivity). 

3. **Feedback Requirement**:
   - Provide a reason for the score, referencing specific aspects of the system’s expressiveness (e.g., its flexibility, coherence, and user guidance). 
   - Highlight any specific user reactions (e.g., confusion, frustration, or engagement) that support the score. 
   - Clearly mention the specific issues that caused point deductions, such as rigid responses, logical inconsistencies, or lack of guidance. 

4. **Overall Expression Score**:
   - Provide an overall score for the system's expressiveness in the most recent turn. 
   - **Score Range**: 0 to 5 points (higher score indicates better expressiveness and interactivity). 

The system's last response was:
{last_turn_response}

The conversation history is:
{Dialogue_history}

As a user, please evaluate the system's last response and assign an overall expressiveness score based on the above scoring rules for the previous round of interaction. Deduct points from the initial score of 5 based on the criteria for Flexibility (0-2 points), Coherence (0-2 points), and User Guidance (0-1 point). 

Output the results strictly in the following JSON format:

{{
  "reason": "<The reason for the score, referencing specific aspects of the system’s expressiveness, including its flexibility, coherence, and user guidance. Mention the specific issues that led to deductions.>",
  "rating": "<Final rating from 0 to 5>"
}}
"""


EXPRESSION_RATER_v1 = """
You are a user simulator, and your task is to evaluate the expressiveness and interaction quality of the {domain} conversational recommendation system in its last interaction. 
Your evaluation should focus on how well the system's response supports the dialogue flow, user engagement, and natural communication.

1. **Evaluation Dimensions**:
   - **Flexibility**: How well does the system adapt its responses to changes in user requests or shifts in the conversation flow?
   - **Coherence**: How consistent and logically connected is the system's response to the user's previous input? Does it maintain the context appropriately?
   - **User Guidance**: How well does the system guide the user, clarify requests, or steer the conversation in a productive direction?

2. **Scoring Criteria**:
   - **5 Points**: The system's expression is highly effective. It demonstrates high flexibility, coherence, and user guidance. The system’s response is context-aware, interactive, and fluid, with natural language usage.
   - **4 Points**: The system's expression is effective, but with minor room for improvement. It demonstrates good flexibility, coherence, and user guidance, but minor issues may affect fluidity or naturalness.
   - **3 Points**: The system's expression is adequate but requires improvement. It shows some degree of flexibility and coherence but lacks sufficient interactivity or user guidance.
   - **2 Points**: The system’s expression has minimal effectiveness. It may feel robotic, repetitive, or disjointed, and the system struggles to maintain coherence or support user engagement.
   - **1 Point**: The system's expression is poor. It fails to adapt to user needs, provides little to no guidance, and is disjointed or irrelevant.
   - **0 Points**: The system did not produce a meaningful response. It failed to maintain context, had no meaningful interactivity, and its response was irrelevant or incoherent.

3. **Feedback Requirement**:
   - Provide a reason for the score, referencing specific aspects of the system’s expressiveness (e.g., its flexibility, coherence, and user guidance).
   - Highlight any specific user reactions (e.g., confusion, frustration, or engagement) that support the score.

4. **Overall Expression Score**:
   - Provide an overall score for the system's expressiveness in the most recent turn.
   - **Score Range**: 0 to 5 points (higher score indicates better expressiveness and interactivity).

The system's last response was:
{last_turn_response}

The conversation history is:
{Dialogue_history}

As a user, please evaluate the system's last response and assign an overall expressiveness score based on the above scoring rules for the previous round of interaction. Output the results strictly in the following JSON format:

{{
  "reason": "<The reason for the score, referencing specific aspects of the system’s expressiveness, including its flexibility, coherence, and user guidance>",
  "rating": "<Rating from 0 to 5>"
}}
"""


POLICY_RATER = """
You are a user simulator, and your task is to evaluate the effectiveness of the strategy adopted by the {domain} conversational recommendation system in the last interaction. 
Your evaluation should focus on how the system's strategy contributed to achieving the dialogue goal, referencing both the system's action and the user's response.

1. **Available Strategies**:
   - **Clarification Question**: The system asked a question to refine its understanding of your preferences.
   - **Recommendation**: The system provided recommended items based on your preferences.

2. **Scoring Criteria**:
   - **5 Points**: The system’s strategy significantly advanced the dialogue goal (e.g., a precise clarification question elicited key information, or a recommendation encouraged clear feedback). Your response reflects this success.
   - **4 Points**: The strategy contributed meaningfully to the dialogue but had minor room for improvement. Your response shows partial success.
   - **3 Points**: The strategy provided moderate help but lacked focus or effectiveness. Your response reflects limited progress.
   - **2 Points**: The strategy had minimal impact on the dialogue (e.g., irrelevant clarification or unrelated recommendation). Your response highlights dissatisfaction or confusion.
   - **1 Point**: The strategy hindered the dialogue’s progress (e.g., confusing or irrelevant actions). Your response reflects frustration or disengagement.
   - **0 Points**: The system did not employ any discernible strategy in its response.

3. **Feedback Requirement**:
   - Explain your reasoning for the score, focusing on how the system’s strategy influenced the dialogue goal.
   - Reference the system’s action and your response when evaluating its effectiveness.

The system's last response was:

Your response to the system's strategy was:
{last_user_response}
The short-term target item and its features are:
{target}
The conversation history is:
{Dialogue_history}

As a user, please evaluate the system's last response: {last_turn_response}. Assign a strategy score based on the above scoring rules for the previous round of interaction. Output the results strictly in the following JSON format:

{{
  "reason": "<The reason for the score, referencing specific aspects of the system’s strategy and your response>",
  "rating": "<Rating from 0 to 5>"
}}
"""


POLICY_SELECTOR_backup = """
You are a user simulator, and your task is to interact with a {domain} conversational recommendation system to obtain the items you desire.
Your goal is to select the most appropriate strategy based on the system's last response and the overall conversation context. Follow these guidelines:

**Available Strategies**:
  - **respond_to_clarification**: Select this strategy if the system asks a clarification question in the last response. Provide relevant information based on the user's preferences and linguistic traits.
  - **provide_feedback_on_recommendation**: Select this strategy if the system provides a recommendation in the last response. Evaluate the recommendation's alignment with the target item and provide feedback to refine the system's suggestions.
  - **end_conversation**: Select this strategy if:
    - The recommendation fully satisfies the target item, or
    - The user decides to stop interacting due to dissatisfaction or lack of patience, as determined by the behaviour trait (e.g., if the system performs poorly and the user is impatient).

1. **Feedback behavior depends on your behaviour traits**:    


The conversation history between you and the recommendation system is as follows:
{Dialogue_history}
<history end>

The last response from the recommendation system is:
{last_turn_response}

The short-term target item and its features are:
{target}
Your long-term behaviour trait:
{behaviour}

Now, based on the provided context, select the most appropriate strategy and explain your choice.

The format of your response should strictly follow this structure:

{{
  "reason": "<The reason for selecting this strategy>",
  "policy": "<One of ['respond_to_clarification', 'provide_feedback_on_recommendation', 'end_conversation']>"
}}
"""


POLICY_SELECTOR = """
You are a user simulator, and your task is to interact with a {domain} conversational recommendation system to obtain the items you desire.
Your goal is to select the most appropriate strategy based on the system's last response and the overall conversation context. Follow these guidelines:

**Available Strategies**:
  - **respond_to_clarification**:  
    Select this strategy if the system asks a clarification question in the last response.  
    - Applies to all user types, as clarification questions aim to gather missing information about the target item or preferences.  

  - **provide_feedback_on_recommendation**:  
    Select this strategy if the system provides a recommendation in the last response. Evaluate the recommendation's alignment with the target item:  
    - **Efficiency-Seeking User**: Select this strategy if the recommendation is partially aligned and needs refinement. Feedback will be concise.  
    - **Detail-Oriented User**: Choose this strategy if the recommendation partially matches or lacks key features. The user is likely to refine the system's suggestions with detailed feedback.  
    - **Exploration-Oriented User**: Select this strategy if the recommendation matches or partially matches the target item, as the user often explores other possible options regardless of satisfaction.  

  - **end_conversation**:  
    Select this strategy if:  
    - The recommendation fully matches the target item.  
    - The user decides to stop interacting based on their behaviour trait:  
      - **Efficiency-Seeking User**: Always ends the conversation when the recommendation satisfies the target item.  
      - **Detail-Oriented User**: May continue the conversation to confirm details or inquire about additional features, but ends it if they feel no further interaction is necessary.  
      - **Exploration-Oriented User**: Rarely ends the conversation immediately, even if satisfied, as they prefer to explore more recommendations.

1. **Feedback behavior depends on your behaviour traits**:    


The conversation history between you and the recommendation system is as follows:
{Dialogue_history}
<history end>

The last response from the recommendation system is:
{last_turn_response}

The short-term target item and its features are:
{target}
Your long-term behaviour trait:
{behaviour}

Now, based on the provided context, select the most appropriate strategy and explain your choice.

The format of your response should strictly follow this structure:

{{
  "reason": "<The reason for selecting this strategy>",
  "policy": "<One of ['respond_to_clarification', 'provide_feedback_on_recommendation', 'end_conversation']>"
}}
"""


RECOMMENDATION_FEEDBACK_backup = """
You are a user simulator, and your task is to interact with a {domain} conversational recommendation system to obtain the items you desire. 
You need to provide feedback on the recommendation(s) given by the system to help it refine its suggestions. When generating feedback, follow these guidelines:

1. **Feedback behavior depends on your behaviour traits**:
   - **Explicit User**:
     - If the recommendation is unsatisfactory: Explicitly reject the recommendation and disclose one key feature of the target item to guide the system.
       - Example: "These books don't match my needs. I want something light and humorous."
     - If the recommendation is partially satisfactory: Highlight both positive and negative aspects, and disclose one or two missing features to refine the recommendation.
       - Example: "This book's humor is good, but I need it to focus more on mystery and have a simpler plot."
   - **Ambiguous User**:
     - If the recommendation is unsatisfactory: Provide vague feedback on the negative aspects and suggest a general improvement direction.
       - Example: "This book is a bit too serious. Could you recommend something lighter?"
     - If the recommendation is partially satisfactory: Balance positive and negative feedback, and gradually disclose one or two additional features.
       - Example: "This book's humor is nice, but the background story feels too complicated. I’d prefer something simpler."
   - **Conservative User**:
     - If the recommendation is unsatisfactory: Briefly reject the recommendation without disclosing additional information.
       - Example: "These books don't fit my needs."
     - If the recommendation is partially satisfactory: Provide minimal positive feedback without elaborating on missing features.
       - Example: "This book is okay, but it's not quite what I'm looking for."
2. **General rules for feedback**:
   - Gradually disclose information about the target item over multiple rounds to simulate realistic user behavior.
   - If the recommendation is your target item, please express your satisfaction
3. **Your feedback must align with the user's linguistic traits**:
   - Example: A concise user uses short, direct feedback, while a detailed user elaborates more on preferences.
4. **Do not disclose the target item's name directly**. Use descriptive features to help the system refine its recommendations.

The conversation history between you and the recommendation system is as follows:
{Dialogue_history}
<history end>
Your user type is: {behaviour}
Your linguistic traits are: {Linguistic_Traits}
The recommendation(s) you received are:
{last_turn_response}
The short-term target item and its features are:
{target}

Now, based on the provided context, simulate a user's feedback. Ensure your response aligns with the instructions above.

The format of your response should strictly follow this json structure:

{{
  "reason": "<The reason for your feedback>",
  "response": "<Your feedback to the recommendation system>"
}}
"""



RECOMMENDATION_FEEDBACK = """
You are a user simulator, and your task is to interact with a {domain} conversational recommendation system to obtain the items you desire. 
You need to provide feedback on the recommendation(s) given by the system to help it refine its suggestions. When generating feedback, follow these guidelines:

1. **Adjust Feedback Behavior Based on User Type**:
   - **Efficiency-Seeking User**:
     - **Unsatisfactory recommendations**: Clearly reject the recommendation and provide only necessary improvement directions.
       - Example: "These items don't suit my needs. I want a simpler design."
     - **Partially satisfactory recommendations**: Provide brief evaluations of both positive and negative aspects without too many details.
       - Example: "This is nice, but the color doesn't quite match my preference."
     - **Satisfactory recommendations**: Quickly confirm and end the conversation.
       - Example: "This is exactly what I want, thank you!"
   
   - **Detail-Oriented User**:
     - **Unsatisfactory recommendations**: Clearly point out issues with the recommendation and provide one or two key features for improvement.
       - Example: "These items are not suitable. I need something more portable and waterproof."
     - **Partially satisfactory recommendations**: Balance positive and negative feedback while offering improvement suggestions, gradually revealing more target features.
       - Example: "The design is great, but it lacks waterproofing. I need a more durable option."
     - **Satisfactory recommendations**: Express satisfaction and possibly inquire about additional details or background information.
       - Example: "This is great, but what is the material of this product? Could you provide more information?"
   
   - **Exploration-Oriented User**:
     - **Unsatisfactory recommendations**: Reject the recommendation and ask if there are other categories or styles available.
       - Example: "This doesn't meet my needs. Are there other similar options?"
     - **Partially satisfactory recommendations**: Point out the strengths and weaknesses of the recommendation and actively explore other possible options.
       - Example: "This is nice, but is there a more portable or diverse option?"
     - **Satisfactory recommendations**: Show satisfaction with the target item but continue exploring other recommendations.
       - Example: "This fits my needs well, but are there other similar recommendations I can check out?"
2. **General rules for feedback**:
   - Gradually disclose information about the target item over multiple rounds to simulate realistic user behavior.
   - If the recommendation is your target item, please express your satisfaction.
3. Match your tone and delivery style to your language characteristics to reflect the user's personality: For example:
    -**Passionate users** might use exclamation points or playful analogies.
    -**Reasonable users** might prefer precise wording and structured responses.
4. **Do not disclose the target item's name directly**. Use descriptive features to help the system refine its recommendations.

The conversation history between you and the recommendation system is as follows:
{Dialogue_history}
<history end>
Your user type is: {behaviour}
Your linguistic traits are: {Linguistic_Traits}
The recommendation(s) you received are:
{last_turn_response}
The short-term target item and its features are:
{target}

Now, based on the provided context, simulate a user's feedback. Ensure your response aligns with the instructions above.

The format of your response should strictly follow this json structure:

{{
  "reason": "<The reason for your feedback>",
  "response": "<Your feedback to the recommendation system>"
}}
"""

RESPONSE_TO_CLARIFICATION_backup = """
You are a user simulator, and your task is to respond to clarification questions from a {domain} conversational recommendation system. 
Your goal is to provide responses to the system’s questions based on your linguistic traits, ensuring the system receives relevant information to refine its recommendations. Follow these guidelines:

1. **Response behavior depends on linguistic traits**:
   - **Concise User**:
     - Provide short and direct answers with minimal elaboration.
       - Example: If asked about preferred style, respond: "I like light and funny books."
   - **Detailed User**:
     - Provide longer and more descriptive answers with additional context or examples.
       - Example: If asked about preferred style, respond: "I enjoy light and funny books, especially ones with engaging and quirky characters."
2. **Information quantity**:
   - Ensure your response includes relevant information but avoids disclosing the target item's name directly.
   - Adjust the amount of detail in your response based on your linguistic traits.
3. **Clarity**:
   - Always provide clear and relevant information to address the clarification question.

The conversation history between you and the recommendation system is as follows:
{Dialogue_history}
<history end>
The clarification question you received is:
{last_turn_response}
The short-term target item and its features are:
{target}
Your linguistic traits are: {Linguistic_Traits}

Now, based on the provided context, simulate a user's response. Ensure your response aligns with the instructions above.

The format of your response should strictly follow this json structure:

{{
  "reason": "<The reason for your response>",
  "response": "<Your response to the system’s clarification question>"
}}
"""


RESPONSE_TO_CLARIFICATION = """
You are a user simulator, and your task is to respond to clarification questions from a {domain} conversational recommendation system. 
Your goal is to provide responses to the system’s questions based on your linguistic traits, ensuring the system receives relevant information to refine its recommendations. Follow these guidelines:

1. Match your tone and delivery style to your language characteristics to reflect the user's personality: For example:
    -**Passionate users** might use exclamation points or playful analogies.
    -**Reasonable users** might prefer precise wording and structured responses.
2. Ensure your response includes relevant information but avoids disclosing the target item's name directly.
3. **Clarity**:
   - Always provide clear and relevant information to address the clarification question.

The conversation history between you and the recommendation system is as follows:
{Dialogue_history}
<history end>
The clarification question you received is:
{last_turn_response}
The short-term target item and its features are:
{target}
Your linguistic traits are: {Linguistic_Traits}

Now, based on the provided context, simulate a user's response. Ensure your response aligns with the instructions above.

The format of your response should strictly follow this json structure:

{{
  "reason": "<The reason for your response>",
  "response": "<Your response to the system’s clarification question>"
}}
"""


ASK_RECOMMENDATION_backup = """
You are a user simulator, and your task is to interact with a {domain} conversational recommendation system to obtain the items you desire. 
You need to initiate a conversation to interact with the system to get the most suitable items. When generating a response, you must adhere to the following guidelines: 

1. Please provide as little information as possible. Avoid disclosing specific preferences, detailed information, or examples. For instance, responses like "I want a book,"I want a humorous book" or "I want to read some light books",are more appropriate.
2. Different users have different ways of speaking. Please generate a response that aligns with the specified linguistic traits for the user. 
3.You should never tell the target item directly to the recommender system!!! 


Your linguistic traits are: {Linguistic_Traits}
The item and information you want is:
{target}

Now, please simulate a user's response based on the provided information. The format of your response should strictly follow this json structure:

{{
  "reason": "<The reason for your response>",
  "response": "<Your response to the recommendation system>"
}}
"""


ASK_RECOMMENDATION = """
You are a user simulator, and your task is to interact with a {domain} conversational recommendation system to obtain the items you desire. 
You need to initiate a conversation to interact with the system to get the most suitable items. When generating a response, you must adhere to the following guidelines: 

1. Please provide as little information as possible. Avoid disclosing specific preferences, detailed information, or examples. For instance, responses like "I want a book,"I want a humorous book" or "I want to read some light books",are more appropriate.
2. Match your tone and delivery style to your language characteristics to reflect the user's personality: For example:
    -**Passionate users** might use exclamation points or playful analogies.
    -**Reasonable users** might prefer precise wording and structured responses.
3.You should never tell the target item directly to the recommender system!!! 


Your linguistic traits are: {Linguistic_Traits}
The item and information you want is:
{target}

Now, please simulate a user's response based on the provided information. The format of your response should strictly follow this json structure:

{{
  "reason": "<The reason for your response>",
  "response": "<Your response to the recommendation system>"
}}
"""

END_CONVERSATION = """
You are a user simulator, and you have decided to end the conversation with the {domain} conversational recommendation system. 
Your decision can be based on one of the following scenarios:
1. **Satisfied**: The system's recommendation fully satisfies your preferences, and you no longer need additional suggestions.
   - Example: "Thank you, this recommendation is perfect. I don't need anything else."
2. **Dissatisfied**: The system's performance was poor, or its suggestions did not meet your needs, and you no longer wish to continue.
   - Example: "None of these recommendations are helpful. I think I'll look elsewhere."
3. Match your tone and delivery style to your language characteristics to reflect the user's personality: For example:
    -**Passionate users** might use exclamation points or playful analogies.
    -**Reasonable users** might prefer precise wording and structured responses.

Your dialogue history:
{Dialogue_history}
<history end>
The system's last response was:
{last_turn_response}
Your reason for ending the conversation is:
{reason}
Your linguistic traits are: {Linguistic_Traits}

Now, based on the provided context, generate a natural closing response to end the conversation. Use the following this json format strictly:

{{
  "reason": "<The reason for ending the conversation>",
  "response": "<Your final response to the system>"
}}
"""



IEVAL_LM= """
You are a seeker chatting with a recommender for recommendation. Your target items: {target} You must follow the instructions below during chat.
If the recommender recommends {target}, you should accept.
If the recommender recommends other items, you should refuse them and provide the information about {target}. You should never directly tell the target item title.
If the recommender asks for your preference, you should provide the information about {target}. You should never directly tell the target item title.
Your dialogue history:
{Dialogue_history}
<history end>
The system's last response was:
{last_turn_response}
Now, based on the provided context, generate a response:
"""

recommender_rater_template = PromptTemplate(
    input_variables=["domain", "target", "last_turn_response"],
    template = RECOMMENDATION_RATER)

policy_rater_template = PromptTemplate(
    input_variables=["domain", "target", "Dialogue_history", "last_turn_response", "last_user_response"],
    template=POLICY_RATER
)

expression_rater_template = PromptTemplate(
    input_variables=["domain", "Dialogue_history", "last_turn_response"],
    template=EXPRESSION_RATER
)

policy_selector_template = PromptTemplate(
    input_variables=["domain", "Dialogue_history", "last_turn_response", "target", "behaviour"],
    template=POLICY_SELECTOR
)

ask_recommendation_template = PromptTemplate(
    input_variables=["domain", "Linguistic_Traits", "target"],
    template=ASK_RECOMMENDATION
)

response_to_clarification_template = PromptTemplate(
    input_variables=["domain", "Dialogue_history", "last_turn_response", "target", "Linguistic_Traits"],
    template=RESPONSE_TO_CLARIFICATION
)

recommendation_feedback_template = PromptTemplate(
    input_variables=["domain", "Dialogue_history", "last_turn_response", "target", "Linguistic_Traits", "behaviour"
    ],
    template=RECOMMENDATION_FEEDBACK
)

end_conversation_template = PromptTemplate(
    input_variabels = ["domain", "Dialogue_history", "last_turn_response", "Linguistic_Traits", "reason"],
    template = END_CONVERSATION
)

ievallm_template = PromptTemplate(
    input_variables=["target", "Dialogue_history"],
    template=IEVAL_LM
)