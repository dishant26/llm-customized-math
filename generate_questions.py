import json
import pandas as pd
import itertools
import random
from typing import List, Dict, Tuple

class QAGenerator:
    def __init__(self, csv_path: str, template_path: str, entity_column: str):
        self.df = pd.read_csv(csv_path)
        with open(template_path, 'r') as f:
            self.templates = json.load(f)
        
        self.entity_column = entity_column
        self.entity_type = entity_column.lower()
        self.columns = [col for col in self.df.columns if col not in [entity_column, 'Rank']]
        self.entities = self.df[entity_column].tolist()

    def get_value(self, entity: str, column: str) -> float:
        return self.df[self.df[self.entity_column] == entity][column].values[0]

    def replace_placeholders(self, template: str, **kwargs) -> str:
        replacements = {
            '<entity>': kwargs.get('entity', ''),
            '<entity1>': kwargs.get('entity1', ''),
            '<entity2>': kwargs.get('entity2', ''),
            '<column>': kwargs.get('column', ''),
            '<n>': str(kwargs.get('n', '')),
            '<entity_type>': self.entity_type,
            '{value}': str(kwargs.get('value', '')),
            '{value1}': str(kwargs.get('value1', '')),
            '{value2}': str(kwargs.get('value2', '')),
            '{winner}': str(kwargs.get('winner', '')),
            '{winning_value}': str(kwargs.get('winning_value', '')),
            '{result}': str(kwargs.get('result', '')),
            '{extreme_type}': str(kwargs.get('extreme_type', '')),
            '{is_ascending}': str(kwargs.get('is_ascending', '')),
            '{ranked_list}': str(kwargs.get('ranked_list', '')),
        }
        
        for placeholder, value in replacements.items():
            if placeholder in template:
                template = template.replace(placeholder, str(value))
        return template

    def generate_direct_recall(self, num_pairs: int) -> List[Dict]:
        qa_pairs = []
        questions = self.templates['direct_recall']['questions']
        
        while len(qa_pairs) < num_pairs:
            question_dict = random.choice(questions)
            entity = random.choice(self.entities)
            column = random.choice(self.columns)
            value = self.get_value(entity, column)
            
            question, answer = self.generate_qa_pair(
                question_dict,
                self.templates['direct_recall']['answer_template'],
                entity=entity,
                column=column,
                value=value
            )
            qa_pairs.append({
                'question': question,
                'answer_steps': answer,
                'category': 'direct_recall',
                'answer_type': question_dict['answer_type']
            })
        return qa_pairs

    def generate_ranking(self, num_pairs: int) -> List[Dict]:
        qa_pairs = []
        questions = self.templates['ranking']['questions']
        
        while len(qa_pairs) < num_pairs:
            question_dict = random.choice(questions)
            column = random.choice(self.columns)
            
            if question_dict['answer_type'] == 'highest/lowest':
                is_highest = random.choice([True, False])
                sorted_df = self.df.sort_values(column, ascending=not is_highest)
                extreme_entity = sorted_df.iloc[0][self.entity_column]
                extreme_value = sorted_df.iloc[0][column]
                
                question, answer = self.generate_qa_pair(
                    question_dict,
                    self.templates['ranking']['answer_template']['highest/lowest'],
                    column=column,
                    extreme_type='highest' if is_highest else 'lowest',
                    is_ascending=str(not is_highest).lower(),
                    entity=extreme_entity,
                    value=extreme_value
                )
            else:  # top_n
                n = random.choice([3, 5])
                sorted_df = self.df.sort_values(column, ascending=False)
                top_n = sorted_df.head(n)
                ranked_list = "\n".join([f"- {row[self.entity_column]}: {row[column]}" 
                                       for _, row in top_n.iterrows()])
                
                question, answer = self.generate_qa_pair(
                    question_dict,
                    self.templates['ranking']['answer_template']['top_n'],
                    column=column,
                    n=n,
                    ranked_list=ranked_list
                )
            
            qa_pairs.append({
                'question': question,
                'answer_steps': answer,
                'category': 'ranking',
                'answer_type': question_dict['answer_type']
            })
        return qa_pairs

    def generate_mathematical(self, num_pairs: int) -> List[Dict]:
        qa_pairs = []
        questions = self.templates['mathematical']['questions']
        
        while len(qa_pairs) < num_pairs:
            question_dict = random.choice(questions)
            column = random.choice(self.columns)
            entity1, entity2 = random.sample(self.entities, 2)
            value1 = self.get_value(entity1, column)
            value2 = self.get_value(entity2, column)
            
            if question_dict['answer_type'] == 'sum_two':
                result = value1 + value2
            elif question_dict['answer_type'] == 'difference':
                result = value1 - value2
            else:  # average_two
                result = (value1 + value2) / 2
            
            question, answer = self.generate_qa_pair(
                question_dict,
                self.templates['mathematical']['answer_template'][question_dict['answer_type']],
                entity1=entity1,
                entity2=entity2,
                column=column,
                value1=value1,
                value2=value2,
                result=round(result, 2)
            )
            qa_pairs.append({
                'question': question,
                'answer_steps': answer,
                'category': 'mathematical',
                'answer_type': question_dict['answer_type']
            })
        return qa_pairs

    def generate_comparison(self, num_pairs: int) -> List[Dict]:
        qa_pairs = []
        questions = self.templates['comparison']['questions']
        
        while len(qa_pairs) < num_pairs:
            question_dict = random.choice(questions)
            column = random.choice(self.columns)
            entity1, entity2 = random.sample(self.entities, 2)
            value1 = self.get_value(entity1, column)
            value2 = self.get_value(entity2, column)
            
            if question_dict['answer_type'] == 'simple_comparison':
                winner = entity1 if value1 > value2 else entity2
                winning_value = max(value1, value2)
                
                question, answer = self.generate_qa_pair(
                    question_dict,
                    self.templates['comparison']['answer_template']['simple_comparison'],
                    entity1=entity1,
                    entity2=entity2,
                    column=column,
                    value1=value1,
                    value2=value2,
                    winner=winner,
                    winning_value=winning_value
                )
            else:  # relative_difference
                diff = abs(value1 - value2)
                base_value = min(value1, value2)
                result = (diff / base_value) * 100
                
                question, answer = self.generate_qa_pair(
                    question_dict,
                    self.templates['comparison']['answer_template']['relative_difference'],
                    entity1=entity1,
                    entity2=entity2,
                    column=column,
                    value1=value1,
                    value2=value2,
                    result=round(result, 2)
                )
            
            qa_pairs.append({
                'question': question,
                'answer_steps': answer,
                'category': 'comparison',
                'answer_type': question_dict['answer_type']
            })
        return qa_pairs

    def generate_qa_pair(self, question_template: dict, answer_template: list, **kwargs) -> Tuple[str, List[str]]:
        question = self.replace_placeholders(question_template['template'], **kwargs)
        answer_steps = [self.replace_placeholders(step, **kwargs) for step in answer_template]
        return question, answer_steps

    def generate_qas(self, pairs_per_category: int = 50) -> List[Dict]:
        qa_pairs = []
        
        # Generate QAs for each category
        qa_pairs.extend(self.generate_direct_recall(pairs_per_category))
        qa_pairs.extend(self.generate_ranking(pairs_per_category))
        qa_pairs.extend(self.generate_mathematical(pairs_per_category))
        qa_pairs.extend(self.generate_comparison(pairs_per_category))
        
        return qa_pairs

    def save_qas(self, qa_pairs: List[Dict], output_file: str):
        with open(output_file, 'w') as f:
            json.dump({
                'qa_pairs': qa_pairs,
                'metadata': {
                    'total_pairs': len(qa_pairs),
                    'pairs_per_category': len(qa_pairs) // 4,
                    'source_file': self.df.name if hasattr(self.df, 'name') else 'unknown',
                    'entity_type': self.entity_type
                }
            }, f, indent=2)

def main():
    generator = QAGenerator(
        csv_path='data/SEC stats - Sheet1.csv',
        template_path='question_templates.json',
        entity_column='Team'
    )
    
    qa_pairs = generator.generate_qas(pairs_per_category=50)
    generator.save_qas(qa_pairs, 'generated_qa_pairs.json')
    
    # Print some samples
    print(f"\nGenerated {len(qa_pairs)} QA pairs")
    print("\nSample QA pairs from each category:")
    for category in ['direct_recall', 'ranking', 'mathematical', 'comparison']:
        category_pairs = [qa for qa in qa_pairs if qa['category'] == category]
        print(f"\n{category.upper()} (Sample):")
        sample = random.choice(category_pairs)
        print(f"Question: {sample['question']}")
        print("Answer steps:")
        for step in sample['answer_steps']:
            print(f"   {step}")

if __name__ == "__main__":
    main() 