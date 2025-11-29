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

from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)

class ReasoningCOT(ReasoningBase):
    """
    Standard Chain-of-Thought Reasoning.
    FIX: Added __init__ to prevent crash.
    """
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str, max_tokens: int = 4096):
        prompt = f'''
{task_description}

Please think step by step to solve this.
'''
        messages = [{"role": "user", "content": prompt}]

        reasoning_result = self.llm(
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens 
        )
        return reasoning_result


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
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str, max_tokens: int = 1000):
        prompt = f'{task_description}'
        messages = [{"role": "user", "content": prompt}]
        return self.llm(messages=messages, temperature=0.1, max_tokens=max_tokens)

class MemoryBase:
    def __init__(self, memory_type: str, llm) -> None:
        self.llm = llm
        try:
            self.embedding = self.llm.get_embedding_model()
        except:
            print("Warning: LLM has no embedding model. Loading local HuggingFace...")
            self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        db_path = os.path.join('./db', memory_type, f'{str(uuid.uuid4())}')
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            
        if self.embedding:
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path
            )

    def __call__(self, current_situation: str = ''):
        if 'review:' in current_situation:
            self.addMemory(current_situation.replace('review:', ''))
        else:
            return self.retriveMemory(current_situation)

class MemoryReflection(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='reflection', llm=llm)

    def retriveMemory(self, query_scenario: str):
        if self.scenario_memory is None or self.scenario_memory._collection.count() == 0:
            return ''

        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=3
        )
        
        insights = [result[0].metadata['insight'] for result in similarity_results]
        
        if not insights:
            return ""

        formatted_memory = "STRATEGIC INSIGHTS FROM PAST USERS:\n"
        for i, insight in enumerate(insights, 1):
            formatted_memory += f"{i}. {insight}\n"
            
        return formatted_memory

    def addMemory(self, current_situation: str):
        # Reflection Step: Extract the "Why"
        prompt = f'''
You have just successfully ranked items for a user.
Review the reasoning below and extract a single "Rule of Thumb" about this user type.
Context:
{current_situation}

Write ONLY the rule (1 sentence).
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
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = RecPlanning(llm=self.llm)
        self.reasoning = ReasoningCOT(profile_type_prompt='', llm=self.llm)
        self.helper_reasoning = RecReasoning(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryReflection(llm=self.llm)

    def my_rating(self, score, additional_indicator=0.0, avg_score=2.5, min_score=0.0, max_score=5.0,
                  base_min=0.0, base_max=5.0, eps=1e-5):
        base_avg = (base_min + base_max) / 2
        adjusted_score = (score - avg_score) / (max_score - min_score + eps) \
            + (score - base_avg) / (base_max - base_min + eps)
        additional_score = math.log(1 + additional_indicator)
        return max(-1, min(1, adjusted_score * (1 + additional_score)))

    def my_user_summary(self, review_details):
        review_texts = []
        for review in review_details:
            review_text = f"Item: {review.get('product_info', '')}\nScore: {review.get('score', 0)}\nReview: {review.get('review_text', '')}"
            review_texts.append(review_text)

        reviews_block = "\n\n".join(review_texts)
        prompt = f'''
Summarize the user's preferences based on these reviews. Focus on specific interests and dislikes.
Reviews:
{reviews_block}
Summary:
'''
        return self.helper_reasoning(prompt)

    def my_product_summary(self, product):
        info = f"{product.get('title', '')} | Category: {product.get('main_category', '')} | {product.get('description', '')}"
        info = info[:400]
        
        prompt = f'''
Summarize the key features of this product in 1-2 sentences:
{info}
Summary:
'''
        return self.helper_reasoning(prompt)

    def workflow(self):
        current_user = self.task['user_id']
        current_candidate_list = self.task['candidate_list']

        retrieved_memory = self.memory(current_situation=f"Recommendation task for User {current_user}")
        if not retrieved_memory:
            retrieved_memory = "No specific past insights found."

        user_info = self.interaction_tool.get_user(current_user)
        user_avg_stars = user_info.get('average_stars', 0)
        user_reviews = self.interaction_tool.get_reviews(user_id=current_user)

        if len(user_reviews) > 0:
            min_user_review_stars = min([review['stars'] for review in user_reviews])
            max_user_review_stars = max([review['stars'] for review in user_reviews])
            user_avg_stars = sum([review['stars'] for review in user_reviews]) / len(user_reviews) if user_avg_stars == 0 else user_avg_stars

            user_review_product_list = [review['item_id'] for review in user_reviews]
            user_review_product_adj_score = [
                self.my_rating(score=review['stars'], min_score=min_user_review_stars, max_score=max_user_review_stars, avg_score=user_avg_stars)
                for review in user_reviews
            ]

            user_review_products = [self.interaction_tool.get_item(item_id=item_id) for item_id in user_review_product_list]
            user_review_product_summaries = [self.my_product_summary(product=product) for product in user_review_products]

            user_review_infos = [
                {'review_text': review['text'], 'score': adj_score, 'product_info': product_summary}
                for review, adj_score, product_summary in zip(user_reviews, user_review_product_adj_score, user_review_product_summaries)
            ]
            user_summary = self.my_user_summary(review_details=user_review_infos)
        else:
            user_summary = "New User. No review history available."

        candidate_product_summaries = []
        concise_candidate_info = []
        
        for item_id in current_candidate_list:
            product = self.interaction_tool.get_item(item_id=item_id)
            product_summary = self.my_product_summary(product=product)
            candidate_product_summaries.append(product_summary)

            entry = (
                f"ID: {item_id}\n"
                f"Title: {product.get('title', 'Unknown')}\n"
                f"Rating: {product.get('average_rating', 'N/A')} ({product.get('ratings_count', 0)} ratings)\n"
                f"Price: {product.get('price', 'N/A')}\n"
                f"Summary: {product_summary}\n"
            )
            concise_candidate_info.append(entry)

        full_candidate_info_block = "\n".join(concise_candidate_info)

        ranking_prompt = f'''
You are a recommender system. Rank the following 20 items based on the user's preferences.

---
### 1. User Profile
{user_summary}

---
### 2. Strategic Insights (Memory)
{retrieved_memory}

---
### 3. Candidate Items
{full_candidate_info_block}

---
### 4. Task
Analyze the items against the user profile.
1. Think step-by-step in the scratchpad about which items match the user's interests and which do not.
2. Output the final ranked list of 20 Item IDs.

<reasoning_scratchpad>
(Step-by-step analysis...)
</reasoning_scratchpad>

<final_ranking>
['Item_ID_1', 'Item_ID_2', ..., 'Item_ID_20']
</final_ranking>
'''

        result = self.reasoning(ranking_prompt, max_tokens=4096)

        if result:
            self.memory(current_situation=f"User Summary: {user_summary}. Reasoning: {result}")

        try:
            pattern = r"<final_ranking>\s*(.*?)\s*</final_ranking>"
            match = re.search(pattern, result, flags=re.DOTALL | re.IGNORECASE)
            
            final_list = []
            if match:
                list_string = match.group(1)
                list_string = list_string.replace("```python", "").replace("```", "").strip()
                final_list = ast.literal_eval(list_string)
            else:
                fallback = re.search(r"\[.*\]", result, flags=re.DOTALL)
                if fallback:
                    final_list = ast.literal_eval(fallback.group(0))

            print('Processed Output:', final_list)
            
            if isinstance(final_list, list) and len(final_list) > 0:
                return final_list
            else:
                print("Output was not a valid list. Returning default.")
                return current_candidate_list

        except Exception as e:
            print(f"Parsing Error: {e}")
            return current_candidate_list