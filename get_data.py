import pandas as pd
import numpy as np
import random
from itertools import combinations
import json

# 设置随机种子
random.seed(42)
np.random.seed(42)

# Golden Rule - 用于生成标签
def check_overdue(features):
    """
    Golden Rule:
    非逾期规则: if Feature_1 <= 100 return False
    逾期规则: if (Feature_5 > 50 and Feature_3 > 10) or Feature_4 > 1: return True
    """
    if features['Feature_1'] <= 100:
        return False
    if (features['Feature_5'] > 50 and features['Feature_3'] > 10) or features['Feature_4'] > 1:
        return True
    return False

# 生成用户特征
def generate_user_features():
    """生成单个用户的特征值"""
    return {
        'Feature_1': random.randint(50, 150),
        'Feature_2': random.randint(5, 20),
        'Feature_3': random.randint(5, 20),
        'Feature_4': random.uniform(0.5, 2.5),
        'Feature_5': random.randint(30, 80)
    }

# 生成三个用户，确保只有一个逾期
def generate_three_users():
    """生成三个用户，确保恰好一个逾期"""
    max_attempts = 100
    for _ in range(max_attempts):
        users = []
        for i in range(3):
            features = generate_user_features()
            overdue = check_overdue(features)
            users.append({
                'name': f'user_{i+1}',
                'features': features,
                'overdue': overdue
            })
        
        # 检查是否恰好一个逾期
        overdue_count = sum([u['overdue'] for u in users])
        if overdue_count == 1:
            return users
    
    # 如果随机生成失败，强制创建一个满足条件的组合
    users = []
    # 创建一个逾期用户
    overdue_user = {
        'name': 'user_1',
        'features': {
            'Feature_1': random.randint(101, 150),
            'Feature_2': random.randint(5, 20),
            'Feature_3': random.randint(11, 20),
            'Feature_4': random.uniform(0.5, 1.0),
            'Feature_5': random.randint(51, 80)
        },
        'overdue': True
    }
    users.append(overdue_user)
    
    # 创建两个非逾期用户
    for i in range(2):
        non_overdue_user = {
            'name': f'user_{i+2}',
            'features': {
                'Feature_1': random.randint(50, 100),
                'Feature_2': random.randint(5, 20),
                'Feature_3': random.randint(5, 20),
                'Feature_4': random.uniform(0.5, 2.5),
                'Feature_5': random.randint(30, 80)
            },
            'overdue': False
        }
        users.append(non_overdue_user)
    
    return users

# 生成条件规则
def generate_conditions(difficulty=1):
    """
    生成不同难度的条件规则
    difficulty: 1-5, 表示条件的数量
    """
    all_conditions = [
        ('Feature_1', '>', 100),
        ('Feature_1', '<=', 100),
        ('Feature_2', '>', 10),
        ('Feature_2', '<=', 10),
        ('Feature_3', '>', 10),
        ('Feature_3', '<=', 10),
        ('Feature_4', '>', 1),
        ('Feature_4', '<=', 1),
        ('Feature_5', '>', 50),
        ('Feature_5', '<=', 50),
    ]
    
    # 随机选择条件数量（不超过difficulty）
    num_conditions = random.randint(1, min(difficulty, len(all_conditions)))
    selected_conditions = random.sample(all_conditions, num_conditions)
    
    return selected_conditions

def check_condition(features, condition):
    """检查单个条件是否满足"""
    feature_name, operator, threshold = condition
    feature_value = features[feature_name]
    
    if operator == '>':
        return feature_value > threshold
    elif operator == '<=':
        return feature_value <= threshold
    elif operator == '>=':
        return feature_value >= threshold
    elif operator == '<':
        return feature_value < threshold
    return False

def evaluate_rule(users, conditions, logic='and', return_predictions=False):
    """
    评估规则是否能正确预测逾期
    logic: 'and' 或 'or'
    return_predictions: 如果为True，返回预测列表；否则返回是否匹配的布尔值
    """
    predictions = []
    for user in users:
        if logic == 'and':
            result = all([check_condition(user['features'], cond) for cond in conditions])
        else:  # or
            result = any([check_condition(user['features'], cond) for cond in conditions])
        predictions.append(result)
    
    if return_predictions:
        return predictions
    
    # 检查是否恰好预测出一个逾期用户，且与实际标签匹配
    actual_overdue = [u['overdue'] for u in users]
    return predictions == actual_overdue

def generate_cot_steps(users, difficulty=2, max_attempts=50):
    """生成思维链步骤"""
    steps = []
    
    for attempt in range(max_attempts):
        conditions = generate_conditions(difficulty)
        logic = random.choice(['and', 'or'])
        
        if evaluate_rule(users, conditions, logic):
            # 生成假设描述
            condition_text = f" {logic} ".join([
                f"{cond[0]} {cond[1]} {cond[2]}" for cond in conditions
            ])
            
            steps.append(f"假设逾期规则为: {condition_text}")
            
            # 为每个用户生成验证步骤
            for user in users:
                feature_desc = ", ".join([
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in user['features'].items()
                ])
                
                matches_rule = evaluate_rule([user], conditions, logic, return_predictions=True)[0]
                actual_overdue = user['overdue']
                
                if matches_rule and actual_overdue:
                    steps.append(f"验证 {user['name']} (特征: {feature_desc}): 满足规则且实际会逾期，预测正确")
                elif not matches_rule and not actual_overdue:
                    steps.append(f"验证 {user['name']} (特征: {feature_desc}): 不满足规则且实际不逾期，预测正确")
                else:
                    steps.append(f"验证 {user['name']} (特征: {feature_desc}): 预测与实际一致")
            
            return steps
    
    # 如果没找到合适的规则，返回基于golden rule的简化版本
    steps.append("假设逾期规则为: Feature_1 > 100 and (Feature_5 > 50 and Feature_3 > 10 or Feature_4 > 1)")
    for user in users:
        if user['overdue']:
            steps.append(f"{user['name']} 满足逾期条件，预测会逾期")
        else:
            steps.append(f"{user['name']} 不满足逾期条件，预测不会逾期")
    
    return steps

def generate_quiz(users):
    """生成问题描述"""
    quiz = "你是一个风控专家，需要判断用户是否会发生逾期。你遇到3个用户: "
    quiz += ", ".join([u['name'] for u in users])
    quiz += "。\n\n"
    
    for user in users:
        quiz += f"{user['name']} 的特征为: "
        feature_items = []
        for k, v in user['features'].items():
            if isinstance(v, float):
                feature_items.append(f"{k}={v:.2f}")
            else:
                feature_items.append(f"{k}={v}")
        quiz += ", ".join(feature_items)
        quiz += "。\n"
    
    quiz += "\n请根据这些特征，推理出哪个用户会发生逾期？"
    return quiz

def generate_sample(difficulty=2):
    """生成一个完整的训练样本"""
    users = generate_three_users()
    
    # 生成各个字段
    names = [u['name'] for u in users]
    solution = [u['overdue'] for u in users]
    
    # 生成solution_text
    solution_texts = []
    for i, user in enumerate(users):
        status = "会逾期" if user['overdue'] else "不会逾期"
        solution_texts.append(f"({i+1}) {user['name']} {status}")
    solution_text = "\n".join(solution_texts)
    
    solution_text_format = solution_text
    
    # 生成quiz
    quiz = generate_quiz(users)
    
    # 生成COT步骤
    cot_head = "让我们一步步思考，通过尝试不同的逾期判定规则，看看哪个规则能够正确识别会逾期的用户。"
    cot_repeat_steps = generate_cot_steps(users, difficulty)
    cot_foot = "通过以上推理，我们找到了能够正确识别逾期用户的规则。"
    
    # 生成statements (特征的结构化表示)
    statements = []
    for user in users:
        statements.append(user['features'])
    
    # 生成prompt
    prompt = [{
        'content': f"""<|im_start|>system
你是一个专业的风控分析助手。助手首先在脑海中思考推理过程，然后为用户提供答案。推理过程和答案分别包含在<think>和<answer>标签中，即<think>推理过程</think> <answer>答案</answer>。
现在用户要求你解决一个逾期预测问题。在思考之后，当你最终得出结论时，请在<answer>标签内清楚地说明每个用户的状态，格式如: (1) user_1 会逾期\\n(2) ... 。
<|im_end|>
<|im_start|>user
{quiz}
<|im_end|>
<|im_start|>assistant
""",
        'role': 'user'
    }]
    
    sample = {
        'quiz': quiz,
        'names': names,
        'solution': solution,
        'solution_text': solution_text,
        'solution_text_format': solution_text_format,
        'cot_head': cot_head,
        'cot_repeat_steps': cot_repeat_steps,
        'cot_foot': cot_foot,
        'statements': statements,
        'prompt': prompt,
        'ability': 'overdue_prediction',
        'data_source': 'generated',
    }
    
    return sample

# 生成训练数据集
def generate_dataset(num_samples=100, difficulty_range=(1, 3)):
    """
    生成训练数据集
    num_samples: 样本数量
    difficulty_range: 难度范围 (min, max)
    """
    dataset = []
    
    for i in range(num_samples):
        difficulty = random.randint(difficulty_range[0], difficulty_range[1])
        sample = generate_sample(difficulty)
        sample['index'] = i
        sample['extra_info'] = {'index': i, 'split': 'train' if i < num_samples * 0.8 else 'test'}
        sample['reward_model'] = {
            'ground_truth': {
                'solution_text_format': sample['solution_text_format'],
                'statements': str(sample['statements'])
            },
            'style': 'rule'
        }
        dataset.append(sample)
        
        if (i + 1) % 10 == 0:
            print(f"已生成 {i + 1}/{num_samples} 个样本")
    
    return dataset

# 主函数
if __name__ == "__main__":
    # 生成数据集
    print("开始生成训练数据...")
    num_samples = 100  # 可以调整样本数量
    dataset = generate_dataset(num_samples=num_samples, difficulty_range=(1, 3))
    
    # 转换为DataFrame
    df = pd.DataFrame(dataset)
    
    # 保存为parquet格式
    output_file = "overdue_training_data.parquet"
    df.to_parquet(output_file, index=False)
    df.to_csv(output_file.replace('.parquet', '.csv'), index=False)
    print(f"\n训练数据已保存到: {output_file}")
    print(f"总样本数: {len(df)}")
    print(f"\n数据集列名: {df.columns.tolist()}")
    
    # 显示前几个样本的示例
    print("\n=== 样本示例 ===")
    print(f"\nQuiz示例:\n{df.iloc[0]['quiz']}")
    print(f"\nSolution: {df.iloc[0]['solution']}")
    print(f"\nSolution Text:\n{df.iloc[0]['solution_text']}")
    print(f"\nCOT Steps: {df.iloc[0]['cot_repeat_steps'][:2]}")  # 只显示前两步
    
    print("\n数据生成完成！")