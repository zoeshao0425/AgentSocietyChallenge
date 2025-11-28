import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import re
import logging
import time
import os
import shutil
import uuid
import ast
import math

# Memory specific imports
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
        a = 0
    return a


class RecPlanning(PlanningBase):
    """Inherits from PlanningBase"""

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """Override the parent class's create_prompt method"""
        if feedback == '':
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}
Task: {task_description}
'''
            prompt = prompt.format(task_description=task_description, task_type=task_type)
        else:
            prompt = '''You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask. Your output format should follow the example below.
The following are some examples:
Task: I need to find some information to complete a recommendation task.
sub-task 1: {{"description": "First I need to find user information", "reasoning instruction": "None"}}
sub-task 2: {{"description": "Next, I need to find item information", "reasoning instruction": "None"}}
sub-task 3: {{"description": "Next, I need to find review information", "reasoning instruction": "None"}}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
'''
            prompt = prompt.format(
                example=few_shot,
                task_description=task_description,
                task_type=task_type,
                feedback=feedback
            )
        return prompt


class RecReasoning(ReasoningBase):
    """Inherits from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str, max_tokens: int = 1000):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}
'''
        prompt = prompt.format(task_description=task_description)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens
        )
        return reasoning_result
    
class ReasoningCOT(ReasoningBase):
    def __call__(self, task_description: str, feedback :str= ''):
        examples, task_description = self.process_task_description(task_description)
        prompt = '''Solve the task step by step. Your instructions must follow the examples.
Here are some examples.
{examples}
Here is the task:
{task_description}'''
        prompt = prompt.format(task_description=task_description, examples=examples)
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
        )
        return reasoning_result
    
class ReasoningRecStepBack(ReasoningBase):
    def __init__(self, profile_type_prompt, llm):
        # Pass memory=None because you handle memory in the workflow
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str, max_tokens: int = 2048):
        # --- OPTIMIZATION 1: Single-Pass Prompting ---
        # Instead of 2 calls, we ask for the Principles AND the Ranking in one go.
        
        prompt = f'''
{task_description}

---
### Your Goal
Rank the 20 Candidate Items based on the User Preference Summary and Past Experiences provided above.

### Instructions
You must follow this exact reasoning process in your output:

1. **Step-Back Analysis (Principles):** - First, analyze the user's history to identify the top 3 "Critical Criteria" (e.g., specific brands, price sensitivity, genres) that matter most to this user.
   
2. **Item Evaluation:**
   - Briefly evaluate how the items fit these Critical Criteria.

3. **Final Ranking:**
   - Output the final sorted list of Item IDs.

### Output Format
<reasoning_scratchpad>
**Critical Criteria:**
1. ...
2. ...
3. ...

**Evaluation:**
(My logic for the top picks vs the bottom picks...)
</reasoning_scratchpad>

<final_ranking>
['Item_ID_1', 'Item_ID_2', ..., 'Item_ID_20']
</final_ranking>
'''
        messages = [{"role": "user", "content": prompt}]
        
        # --- OPTIMIZATION 2: Respect the max_tokens argument ---
        # Your previous code ignored the max_tokens passed from workflow
        reasoning_result = self.llm(
            messages=messages, 
            temperature=0.1, 
            max_tokens=max_tokens
        )
        
        return reasoning_result

class MemoryBase:
    def __init__(self, memory_type: str, llm) -> None:
        """
        Initialize the memory base class
        
        Args:
            memory_type: Type of memory
            llm: LLM instance used to generate memory-related text
        """
        self.llm = llm
        self.embedding = self.llm.get_embedding_model()
        db_path = os.path.join('./db', memory_type, f'{str(uuid.uuid4())}')
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )

    def __call__(self, current_situation: str = ''):
        if 'review:' in current_situation:
            self.addMemory(current_situation.replace('review:', ''))
        else:
            return self.retriveMemory(current_situation)

    def retriveMemory(self, query_scenario: str):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def addMemory(self, current_situation: str):
        raise NotImplementedError("This method should be implemented by subclasses.")

class MemoryDILU(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='dilu', llm=llm)

    def retriveMemory(self, query_scenario: str):
        task_name = query_scenario
        if self.scenario_memory._collection.count() == 0:
            return ''

        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)

        task_trajectories = [
            result[0].metadata['task_trajectory'] for result in similarity_results
        ]
        
        return '\n'.join(task_trajectories)

    def addMemory(self, current_situation: str):
        task_name = current_situation

        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        self.scenario_memory.add_documents([memory_doc])
        
class MemoryReflection(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='reflection', llm=llm)

    def retriveMemory(self, query_scenario: str):
        """
        Retrieves high-level insights/principles applicable to the current user.
        """
        if self.scenario_memory._collection.count() == 0:
            return ''

        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=3
        )
        
        insights = [result[0].metadata['insight'] for result in similarity_results]
        
        if not insights:
            return ""

        formatted_memory = "Reflections from similar past users:\n"
        for i, insight in enumerate(insights, 1):
            formatted_memory += f"{i}. {insight}\n"
            
        return formatted_memory

    def addMemory(self, current_situation: str):
        """
        Instead of saving the raw text, we ask the LLM to reflect on WHY it worked
        and save that insight.
        """

        prompt = f'''
You have just completed a recommendation task successfully. 
Review the task details and your reasoning process below.
Extract a general "Rule of Thumb" or "Insight" that explains why this recommendation strategy worked. 
This insight should be useful for future tasks with similar users.

Input Context:
{current_situation}

Write ONLY the insight (1-2 sentences).
Insight:
'''

        insight = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1)
        memory_doc = Document(
            page_content=insight, 
            metadata={
                "insight": insight,
                "full_trajectory": current_situation
            }
        )
        
        self.scenario_memory.add_documents([memory_doc])

class MyRecommendationAgent(RecommendationAgent):
    """ Participant's implementation of SimulationAgent """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = ReasoningRecStepBack(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryReflection(llm=self.llm)

    def my_rating(self, score, additional_indicator=0.0, avg_score=2.5, min_score=0.0, max_score=5.0,
                  base_min=0.0, base_max=5.0, eps=1e-5):
        base_avg = (base_min + base_max) / 2
        adjusted_score = (score - avg_score) / (max_score - min_score + eps) \
            + (score - base_avg) / (base_max - base_min + eps)
        additional_score = math.log(1 + additional_indicator)
        # clip between -1 and 1
        return max(-1, min(1, adjusted_score * (1 + additional_score)))

    def my_user_summary(self, review_details):
        review_texts = []
        for review in review_details:
            review_text = f'''Review Text: {review.get('review_text', '')}
Score: {review.get('score', 0)}
Product Info: {review.get('product_info', '')}
'''
            review_texts.append(review_text)

        reviews_block = "\n\n".join(review_texts)
        prompt = f'''
Summarize the preferences of the user from the following reviews on different products into a concise user preference description, focusing on key aspects such as preferences, dislikes, and notable patterns. The scores for the reviews are between [-1, 1], from hate to like. Keep the summary brief and to the point.
Here are the reviews:
{reviews_block}
--- End of Reviews ---
Your summary:
'''
        return self.reasoning(prompt)

    def my_product_summary(self, product):
        product_info = {
            key: product[key]
            for key in [
                'item_id', 'name', 'stars', 'review_count', 'attributes', 'title',
                'average_rating', 'rating_number', 'description', 'ratings_count',
                'title_without_series', 'details', 'main_category', 'categories'
            ]
            if key in product
        }
        product_info_block = "\n".join(
            [f"{key}: {value}" for key, value in product_info.items()]
        )
        prompt = f'''
Summarize the key features and attributes of the following product into a concise description, focusing on aspects such as quality, usability, and unique selling points. Keep the summary brief and to the point. And the rating score is between [0, 5].
Here is the product information:
{product_info_block}
--- End of Product Information ---
Your summary:
'''
        return self.reasoning(prompt)

    def workflow(self):
        """
        Simulate user behavior
        Returns:
            list: Sorted list of item IDs
        """
        current_user = self.task['user_id']
        current_candidate_list = self.task['candidate_list']

        retrieved_memory = self.memory(current_situation=f"Recommendation task for User {current_user}")
        
        if not retrieved_memory:
            retrieved_memory = "No past similar tasks found."

        user_info = self.interaction_tool.get_user(current_user)
        user_avg_stars = user_info.get('average_stars', 0)
        user_friends = user_info.get('friends', '')
        user_friends = user_friends.split(',') if user_friends != '' else []

        user_reviews = self.interaction_tool.get_reviews(user_id=current_user)

        if len(user_reviews) > 0:
            min_user_review_stars = min([review['stars'] for review in user_reviews])
            max_user_review_stars = max([review['stars'] for review in user_reviews])
            user_avg_stars = sum([review['stars'] for review in user_reviews]) / len(user_reviews) if user_avg_stars == 0 else user_avg_stars

            user_review_product_list = [review['item_id'] for review in user_reviews]
            user_review_product_adj_score = [
                self.my_rating(
                    score=review['stars'],
                    min_score=min_user_review_stars,
                    max_score=max_user_review_stars,
                    avg_score=user_avg_stars
                )
                for review in user_reviews
            ]

            user_review_products = [
                self.interaction_tool.get_item(item_id=item_id)
                for item_id in user_review_product_list
            ]

            user_review_product_summaries = [
                self.my_product_summary(product=product)
                for product in user_review_products
            ]

            user_review_infos = [
                {
                    'review_text': review['text'],
                    'score': adj_score,
                    'product_info': product_summary
                }
                for review, adj_score, product_summary in zip(
                    user_reviews,
                    user_review_product_adj_score,
                    user_review_product_summaries
                )
            ]

            user_summary = self.my_user_summary(review_details=user_review_infos)
        else:
            user_summary = "The user has no reviews."

        # process friends' information
        if len(user_friends) > 0:
            friend_summaries = []
            for friend_id in user_friends:
                friend_info = self.interaction_tool.get_user(user_id=friend_id)
                if friend_info is None:
                    continue

                friend_avg_stars = friend_info.get('average_stars', 0)
                friend_reviews = self.interaction_tool.get_reviews(user_id=friend_id)
                if len(friend_reviews) == 0:
                    continue

                min_friend_review_stars = min([review['stars'] for review in friend_reviews])
                max_friend_review_stars = max([review['stars'] for review in friend_reviews])
                friend_avg_stars = sum([review['stars'] for review in friend_reviews]) / len(friend_reviews) if friend_avg_stars == 0 else friend_avg_stars

                friend_review_product_adj_score = [
                    self.my_rating(
                        score=review['stars'],
                        min_score=min_friend_review_stars,
                        max_score=max_friend_review_stars,
                        avg_score=friend_avg_stars
                    )
                    for review in friend_reviews
                ]

                friend_review_products = [
                    self.interaction_tool.get_item(item_id=item_id)
                    for item_id in [review['item_id'] for review in friend_reviews]
                ]

                friend_review_product_summaries = [
                    self.my_product_summary(product=product)
                    for product in friend_review_products
                ]

                friend_review_infos = [
                    {
                        'review_text': review['text'],
                        'score': adj_score,
                        'product_info': product_summary
                    }
                    for review, adj_score, product_summary in zip(
                        friend_reviews,
                        friend_review_product_adj_score,
                        friend_review_product_summaries
                    )
                ]

                friend_summary = self.my_user_summary(review_details=friend_review_infos)
                friend_summaries.append(friend_summary)
        else:
            friend_summaries = []

        if len(friend_summaries) > 0:
            friends_block = "\n\n".join(friend_summaries)
            combined_summary_prompt = f'''
You are given the following user's preference summary and their friends' preference summaries. The user's preferences may be influenced by their friends, but they are ultimately their own. Adjust the user's preference summary by incorporating relevant aspects from their friends' summaries, while maintaining the core of the user's original preferences. Keep the adjusted summary concise and to the point.
The user's summary is as follows:
{user_summary}
The friends' summaries are as follows:
{friends_block}
--- End of Summaries ---
Your adjusted user summary:
'''
            adjusted_user_summary = self.reasoning(combined_summary_prompt)
        else:
            adjusted_user_summary = user_summary

        candidate_product_summaries = []
        candidate_dicts = []
        for item_id in current_candidate_list:
            product = self.interaction_tool.get_item(item_id=item_id)
            product_summary = self.my_product_summary(product=product)
            candidate_dicts.append(product)
            candidate_product_summaries.append(product_summary)

        candidate_info_strs = []
        for i in range(len(current_candidate_list)):
            candidate_info_str = (
                f"Item ID: {current_candidate_list[i]}\n"
                f"Summary: {candidate_product_summaries[i]}\n"
            )
            candidate_short_info = {
                key: candidate_dicts[i][key]
                for key in [
                    'item_id', 'name', 'stars', 'review_count', 'attributes',
                    'title', 'average_rating', 'rating_number', 'description',
                    'ratings_count', 'title_without_series', 'details',
                    'main_category', 'categories'
                ]
                if key in candidate_dicts[i]
            }
            candidate_info_str += "\n".join(
                [f"{key}: {value}" for key, value in candidate_short_info.items()]
            )
            candidate_info_strs.append(candidate_info_str)

        full_candidate_info_block = "\n\n".join(candidate_info_strs)

        ranking_prompt = f'''
You are a real user on an online platform. Your task is to rank a list of 20 items based on your specific preferences.

---
### 1. Your Preference Summary
This is your persona. All your ranking decisions must be based on these interests.
{adjusted_user_summary}

---
### 2. Relevant Past Experiences (Memory)
Here are notes from similar tasks you have completed in the past. Use these to guide your reasoning strategy:
{retrieved_memory}

---
### 3. Candidate Items
You must rank ALL 20 of the following item IDs:
{self.task['candidate_list']}

---
### 4. Your Task & Reasoning Process
You must follow this exact two-step reasoning process. Write all your thoughts inside the <reasoning_scratchpad> block.

**Step 1: Initial Analysis (Summaries)**
First, read the **Candidate Product Summaries**. Compare them to your **Preference Summary**. In the scratchpad, note your initial impressions:
* Which items seem like a strong match?
* Which items seem like a weak match or irrelevant?
* Why? (e.g., "Item 'A' matches my interest in X, Item 'B' does not.")

**Candidate Product Summaries:**
{candidate_product_summaries}

**Step 2: Detailed Refinement (Full Info)**
Next, read the **Full Candidate Information**. Use these details to confirm or change your initial impressions from Step 1. In the scratchpad, refine your thoughts:
* Did an item that looked good in the summary turn out to be bad (or vice-versa)?
* Product with better ratings may be ranked higher, more rating counts might mean more popular, but the most important is the match to your preferences.
* Use this full info to build your final, complete ranking from 1 to 20.

**Full Candidate Information:**
{full_candidate_info_block}

---
### 5. Final Output
Provide your response in the *exact* format below.
1. First, fill in the <reasoning_scratchpad> with your step-by-step analysis from Steps 1 and 2.
2. Second, provide the final ranked list inside the <final_ranking> block. This block must ONLY contain the Python list of 20 item IDs.

<reasoning_scratchpad>
**Step 1 (Summary Analysis):**
(My preferences are... Based on these, the summaries for items [...] look highly relevant because... Items [...] seem irrelevant because...)

**Step 2 (Full Info Refinement):**
(After reading the full details, Item 'X' is definitely my top pick because... Item 'Y', which looked good in the summary, is actually not what I want because... My final ordering logic is...)
</reasoning_scratchpad>

<final_ranking>
{self.task['candidate_list']}
</final_ranking>
'''

        result = self.reasoning(ranking_prompt, max_tokens=12800)

        if result:
            self.memory(current_situation=f"review:Task User {current_user} completed. Reasoning: {result}")

        try:
            pattern = r"<final_ranking>\s*(.*?)\s*</final_ranking>"
            match = re.search(pattern, result, flags=re.DOTALL | re.IGNORECASE)
            if not match:
                print("Error: Could not find <final_ranking> tags in the output.")
                return None

            list_string = match.group(1)
            final_list = ast.literal_eval(list_string)

            print('Processed Output:', final_list)
            if isinstance(final_list, list):
                return final_list
            else:
                print(f"Error: Extracted content was not a list. Found type: {type(final_list)}")
                return None

        except (ValueError, SyntaxError) as e:
            print(f"Error: Failed to parse the list string. Details: {e}")
            print(f"Raw string that failed: {list_string}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None