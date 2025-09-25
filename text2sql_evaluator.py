import json
import os
from typing import Dict, Optional

import sqlparse
from openai import OpenAI

from bm25_question_retriever import BM25QuestionRetriever

client = OpenAI(
    api_key=os.getenv("ALI_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
bm25_query = BM25QuestionRetriever("q2sql_pairs.json")


def normalize_sql(sql: str) -> str:
    """
    标准化SQL语句：统一关键字大小写和格式[1](@ref)。
    """
    try:
        parsed = sqlparse.parse(sql)
        if not parsed:
            return sql.strip().upper()
        formatted = sqlparse.format(
            sql,
            reindent=True,
            keyword_case='upper',
            strip_whitespace=True
        )
        return formatted.strip()
    except Exception as e:
        print(f"SQL标准化失败: {e}")
        return sql.strip().upper()


def evaluate_faithfulness(question: str, generated_sql: str) -> float:
    """
    忠实度评估：检查生成的SQL是否基于问题意图和数据库上下文，避免幻觉[3,10](@ref)。
    返回得分（0-1），1表示完全忠实。
    """

    prompt = f"""
    给定一个自然语言问题和一个SQL查询，判断该SQL是否忠实地反映了问题的意图。
    问题: {question}
    SQL: {generated_sql}
    要求:
    - 如果SQL完全符合问题要求（如查询字段、条件、表关系正确），返回1。
    - 如果SQL部分偏离问题意图（如多余或缺失条件），返回0.5。
    - 如果SQL与问题无关或严重错误，返回0。
    仅返回数字得分（0/0.5/1），不要额外解释。
    """

    try:
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))  # 确保得分在0-1范围内
    except Exception as e:
        print(f"忠实度评估失败: {e}")
        return 0.0


def evaluate_answer_relevancy(question: str, generated_sql: str) -> float:
    """
    答案相关性评估：衡量生成的SQL与问题的语义相关性[4,5](@ref)。
    返回得分（0-1），1表示完全相关。
    """

    prompt = f"""
    评估以下SQL查询与自然语言问题的相关性：
    问题: {question}
    SQL: {generated_sql}
    要求:
    - 如果SQL能准确回答问题（如返回结果直接对应问题需求），返回1。
    - 如果SQL部分回答问题（如结果需二次处理才能使用），返回0.5。
    - 如果SQL无法回答问题或逻辑无关，返回0。
    仅返回数字得分。
    """

    try:
        response = client.chat.completions.create(
            model="qwen3-max",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"答案相关性评估失败: {e}")
        return 0.0


def get_standard_sql(question: str) -> Optional[str]:
    """从数据集中获取问题的标准SQL答案"""
    std_question, std_sql, score = bm25_query.retrieve_similar_question(question)
    if score <= 0.7:
        return ""
    return std_sql


def evaluate_exact_match(question: str, generated_sql: str) -> bool:
    """
    精确匹配评估（传统方法）：比较生成的SQL与标准答案是否完全一致[5](@ref)。
    """
    standard_sql = get_standard_sql(question)
    if standard_sql is None:
        return False

    norm_gen = normalize_sql(generated_sql)
    norm_std = normalize_sql(standard_sql)
    return norm_gen == norm_std


def evaluate_single(question: str, generated_sql: str) -> Dict[str, float]:
    """
    综合评估单个样本的多维度指标。
    """
    metrics = {
        "exact_match": evaluate_exact_match(question, generated_sql),
        "faithfulness": evaluate_faithfulness(question, generated_sql),
        "answer_relevancy": evaluate_answer_relevancy(question, generated_sql)
    }
    # 计算综合得分（加权平均）
    weights = {"exact_match": 0.4, "faithfulness": 0.3, "answer_relevancy": 0.3}  # 可调整权重
    metrics["overall_score"] = sum(metrics[k] * weights[k] for k in weights)
    return metrics


class Text2SQLRAGEvaluator:
    """
    基于RAG框架的Text2SQL评估器，支持多维度指标评估。
    评估指标包括：精确匹配（EM）、忠实度（Faithfulness）、答案相关性（Answer Relevancy）。
    """

    def __init__(self, dataset_path: str):
        """
        初始化评估器，加载数据集并配置LLM。

        Args:
            dataset_path (str): JSON数据集文件路径，包含"question"和"sql"字段。
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)


if __name__ == "__main__":
    evaluator = Text2SQLRAGEvaluator("q2sql_pairs.json")
    question = "请执行操作：创建类别 Horror "
    generated_sql = "INSERT INTO category (name) VALUES ('Horror');"  # 模型生成的SQL
    results = evaluate_single(question, generated_sql)
    print(f"评估结果: {results}")
