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
    
class ReasoningBatchScoring(ReasoningBase):
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, user_profile, memory, batch_items):
        """
        Scores a small batch of items (e.g., 5) against the profile.
        """
        items_text = ""
        for item in batch_items:
            items_text += f"ID: {item['id']}\nDetails: {item['details']}\n\n"

        prompt = f'''
{user_profile}

{memory}

---
### TASK: Rate Candidate Items
Evaluate the following items against the User Profile above.
Assign a **Relevance Score (0-10)** to EACH item, where 10 is a perfect match and 0 is irrelevant.

CANDIDATE ITEMS:
{items_text}

### OUTPUT FORMAT
You must output exactly one line per item in this format:
ID: [Item_ID] | Score: [Number]

Example:
ID: B0012345 | Score: 8.5
'''
        messages = [{"role": "user", "content": prompt}]
        # High temp allowed for scoring nuance, but keep it controlled
        result = self.llm(messages=messages, temperature=0.1, max_tokens=1000)
        return result

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
        self.scorer = ReasoningBatchScoring(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryReflection(llm=self.llm)

    def my_rating(self, score, additional_indicator=0.0, avg_score=2.5, min_score=0.0, max_score=5.0,
                  base_min=0.0, base_max=5.0, eps=1e-5):
        base_avg = (base_min + base_max) / 2
        adjusted_score = (score - avg_score) / (max_score - min_score + eps) \
            + (score - base_avg) / (base_max - base_min + eps)
        additional_score = math.log(1 + additional_indicator)
        return max(-1, min(1, adjusted_score * (1 + additional_score)))

    def my_user_summary(self, review_details):
        review_texts = "\n".join([f"Item: {r['product_info']}\nUser Score: {r['score']}\nReview: {r['review_text']}" for r in review_details])
        
        prompt = f'''
Analyze the following reviews to build a structured User Persona.
Reviews:
{review_texts}

Output strictly in this format:
**Top 3 Interests:** [Interest 1], [Interest 2], [Interest 3]
**Top 3 Dealbreakers:** [Dislike 1], [Dislike 2], [Dislike 3]
**Price Sensitivity:** [High/Medium/Low]
'''
        return self.scorer.llm(messages=[{"role": "user", "content": prompt}], temperature=0.1)

    def my_product_summary(self, product):
        info = f"{product.get('title', '')} {product.get('categories', '')} {product.get('description', '')}"
        return info[:500] 

    def workflow(self):
        current_user = self.task['user_id']
        current_candidate_list = self.task['candidate_list']

        retrieved_memory = self.memory(current_situation=f"User {current_user}")
        if not retrieved_memory:
            retrieved_memory = ""

        user_reviews = self.interaction_tool.get_reviews(user_id=current_user)
        user_info = self.interaction_tool.get_user(current_user)
        
        if user_reviews:
            user_avg = user_info.get('average_stars', 0) or 2.5
            user_review_infos = []
            for review in user_reviews:
                prod = self.interaction_tool.get_item(review['item_id'])
                summary = self.my_product_summary(prod)
                score = self.my_rating(review['stars'], avg_score=user_avg, max_score=5.0)
                user_review_infos.append({'review_text': review['text'], 'score': score, 'product_info': summary})
            
            user_profile = self.my_user_summary(user_review_infos)
        else:
            user_profile = "New User. No specific history. Assume general popularity."

        candidate_data = []
        for item_id in current_candidate_list:
            item = self.interaction_tool.get_item(item_id)
            details = (
                f"Title: {item.get('title', 'Unknown')}\n"
                f"Category: {item.get('main_category', 'N/A')}\n"
                f"Rating: {item.get('average_rating', 'N/A')} ({item.get('ratings_count', 0)})\n"
                f"Price: {item.get('price', 'N/A')}"
            )
            candidate_data.append({'id': item_id, 'details': details})

        batch_size = 5
        scored_results = []
        
        for i in range(0, len(candidate_data), batch_size):
            batch = candidate_data[i : i + batch_size]

            llm_output = self.scorer(
                user_profile=user_profile, 
                memory=retrieved_memory, 
                batch_items=batch
            )
            
            for item in batch:
                item_id = item['id']
                score = 0.0
                try:
                    pattern = re.compile(re.escape(item_id) + r".*?Score:\s*([\d\.]+)", re.IGNORECASE | re.DOTALL)
                    match = pattern.search(llm_output)
                    if match:
                        score = float(match.group(1))
                except:
                    pass
                
                scored_results.append((item_id, score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        final_ranking = [x[0] for x in scored_results]

        if len(final_ranking) < len(current_candidate_list):
            missing = [x for x in current_candidate_list if x not in final_ranking]
            final_ranking.extend(missing)

        print(f"Final Ranking: {final_ranking}")

        top_item = final_ranking[0]
        self.memory(current_situation=f"User Profile: {user_profile}. \nTop Recommendation: {top_item}. \nReason: High match score based on batch analysis.")

        return final_ranking