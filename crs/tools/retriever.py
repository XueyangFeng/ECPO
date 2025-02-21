import faiss
import json
import numpy as np

class Retriever:
    def __init__(self, index_file, metadata_file, model):
        """
        初始化 Retriever 类
        
        参数:
        - index_file: 索引文件路径
        - metadata_file: 元数据文件路径
        - model: 用于生成嵌入的模型
        """
        self.index = self._load_index(index_file)
        self.items = self._load_metadata(metadata_file)
        self.model = model
    
    def _load_index(self, index_file):
        """
        加载 FAISS 索引
        """
        return faiss.read_index(index_file)
    
    def _load_metadata(self, metadata_file):
        """
        加载元数据文件
        """
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def query(self, query_text, top_k=5):
        """
        查询索引，返回最近邻结果
        
        参数:
        - query_text: 查询文本
        - top_k: 返回的最近邻数量，默认为 5
        
        返回:
        - results: 包含最近邻结果的列表，每个结果包括物品信息和距离
        """
        # 对查询文本生成嵌入
        query_embedding = self.model.encode([query_text])
        # 查询索引
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        # 构造返回结果
        results = [{"Item": self.items[idx], "Distance": distances[0][i]} for i, idx in enumerate(indices[0])]
        return results
