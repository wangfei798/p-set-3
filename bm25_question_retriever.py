import json
import re
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi


def _tokenize_text(text: str) -> List[str]:
    """中英文混合分词，过滤标点符号"""
    filtered_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    tokens = jieba.lcut(filtered_text, cut_all=False)  # 精确模式分词
    return [token.strip() for token in tokens if token.strip()]


def _tokenize_corpus(questions: List[str]) -> List[List[str]]:
    """对全部标准问题进行分词"""
    return [_tokenize_text(q) for q in questions]


class BM25QuestionRetriever:
    """
    基于BM25的问题检索器，用于匹配用户问题与数据集中的标准问题。
    """

    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        # 预处理：构建标准问题库和对应的SQL答案
        self.questions = [item['question'] for item in self.dataset]
        self.sql_answers = {item['question']: item['sql'] for item in self.dataset}
        # 分词并构建BM25索引
        self.tokenized_questions = _tokenize_corpus(self.questions)
        self.bm25_index = BM25Okapi(self.tokenized_questions)

    def retrieve_similar_question(self, user_question: str, top_k: int = 3) -> Tuple[str, str, float]:
        """
        检索最相似的标准问题及对应的SQL答案。
        Args:
            user_question: 用户输入的问题
            top_k: 返回Top-K个最相似结果
        Returns:
            tuple: (匹配的标准问题, 对应的SQL答案, BM25得分)
        """
        tokenized_query = _tokenize_text(user_question)
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # 选择得分最高的结果
        best_index = top_indices[0]
        best_question = self.questions[best_index]
        best_sql = self.sql_answers[best_question]
        best_score = scores[best_index]
        return best_question, best_sql, best_score
