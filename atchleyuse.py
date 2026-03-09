import numpy as np
import os
import pandas as pd
import chardet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

class DiffPool(nn.Module):
    """完全动态的DiffPool，适应任意节点数 - 改进版，考虑节点度和节点类型"""
    def __init__(self, node_dim, edge_dim, ratio=0.5):
        super(DiffPool, self).__init__()
        self.ratio = ratio
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # 节点特征处理器
        self.gnn = EfficientGraphConv(node_dim, edge_dim)
        
        # 改进的分配分数生成器：考虑节点特征、度和类型
        # 输入: 节点特征(node_dim) + 度中心性(1) + 节点类型(1) = node_dim+2
        self.assignment_net = nn.Sequential(
            nn.Linear(node_dim + 2, (node_dim + 2) * 2),  # 增加输入维度
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear((node_dim + 2) * 2, node_dim + 2),
            nn.LeakyReLU(0.1),
            nn.Linear(node_dim + 2, node_dim),  # 输出到节点特征维度
            nn.LeakyReLU(0.1)
        )
        
        # 可学习的簇原型向量
        self.cluster_prototypes = nn.Parameter(torch.randn(1, node_dim))
    
    def compute_degree_centrality(self, x, edge_index):
        """计算度中心性"""
        num_nodes = x.size(0)
        device = x.device
        degree = torch.zeros(num_nodes, device=device)
        
        if edge_index.size(1) > 0:
            for i in range(edge_index.size(1)):
                src = edge_index[0, i]
                dst = edge_index[1, i]
                if src < num_nodes:
                    degree[src] += 1
                if dst < num_nodes:
                    degree[dst] += 1
        
        # 归一化
        if degree.max() > 0:
            degree = degree / degree.max()
        
        return degree.unsqueeze(1)  # [num_nodes, 1]
    
    def forward(self, x, edge_index, edge_attr, node_types=None):
        """
        改进的forward函数，考虑节点度和类型
        """
        if x.size(0) == 0:
            return x, torch.zeros((2, 0), dtype=torch.long, device=x.device), \
                   torch.zeros((0, self.edge_dim), device=x.device), \
                   torch.zeros((0, 0), device=x.device)
        
        num_nodes = x.size(0)
        k = max(1, int(num_nodes * self.ratio))
        
        # 更新节点特征
        x = self.gnn(x, edge_index, edge_attr)
        
        # 计算度中心性
        degree_centrality = self.compute_degree_centrality(x, edge_index)
        
        # 确保节点类型格式正确
        if node_types is None:
            node_types = torch.zeros((num_nodes, 1), device=x.device)
        elif node_types.dim() == 1:
            node_types = node_types.unsqueeze(1)
        
        # 拼接节点特征、度中心性和节点类型
        combined_features = torch.cat([x, degree_centrality, node_types], dim=1)
        
        # 为每个节点生成分配表示（考虑节点特征、度和类型）
        node_assign_features = self.assignment_net(combined_features)
        
        # 动态生成k个簇原型
        cluster_proto = self.cluster_prototypes.expand(k, -1).to(x.device)
        
        # 计算节点到簇的相似度
        similarity = torch.matmul(node_assign_features, cluster_proto.t())
        
        # 应用softmax得到分配矩阵
        A = F.softmax(similarity, dim=1)
        
        # 池化特征
        new_x = torch.matmul(A.t(), x)
        
        # 计算新边索引和新节点类型
        new_edge_index, new_edge_attr, new_node_types = self.compute_new_graph(
            A, edge_index, edge_attr, node_types
        )
        
        return new_x, new_edge_index, new_edge_attr, A, new_node_types
    
    def compute_new_graph(self, A, edge_index, edge_attr, node_types):
        """计算新图和节点类型"""
        # Map old nodes to new clusters
        node_to_cluster = torch.argmax(A, dim=1)  # [num_nodes]
        
        new_edge_index = []
        new_edge_attr = []
        
        for i in range(edge_index.size(1)):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            
            if src >= node_to_cluster.size(0) or dst >= node_to_cluster.size(0):
                continue
                
            new_src = node_to_cluster[src]
            new_dst = node_to_cluster[dst]
            if new_src != new_dst:
                new_edge_index.append([new_src, new_dst])
                new_edge_attr.append(edge_attr[i])
        
        if len(new_edge_index) > 0:
            new_edge_index = torch.tensor(new_edge_index, dtype=torch.long, device=A.device).t()
            new_edge_attr = torch.stack(new_edge_attr)
        else:
            new_edge_index = torch.zeros((2, 0), dtype=torch.long, device=A.device)
            new_edge_attr = torch.zeros((0, self.edge_dim), device=A.device)
        
        # 更新节点类型：通过加权平均
        new_node_types = torch.matmul(A.t(), node_types)
        
        return new_edge_index, new_edge_attr, new_node_types

class EfficientGraphConv(nn.Module):
    """高效图卷积层 - 边特征与节点特征点乘"""
    def __init__(self, node_dim=8, edge_dim=7, use_atchley_transform=True):
        super(EfficientGraphConv, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.use_atchley_transform = use_atchley_transform
        
        # Atchley特征转换：5维 → 8维（添加隐藏层）
        if use_atchley_transform:
            self.atchley_transform = nn.Sequential(
                nn.Linear(5, node_dim * 2),  # 第一个隐藏层
                nn.LeakyReLU(0.1),
                nn.Dropout(0.05),
                nn.Linear(node_dim * 2, node_dim)  # 输出层
            )
        
        # 边特征转换：7维 → 8维（添加隐藏层）
        self.edge_feature_transform = nn.Sequential(
            nn.Linear(edge_dim, node_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(node_dim, node_dim)
        )
        
        # 节点特征变换
        self.node_transform = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(node_dim * 2, node_dim)
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.ones(1))
        
        # 点乘缩放参数
        self.dot_product_scale = nn.Parameter(torch.ones(1))
        
        # 轻微dropout
        self.dropout = nn.Dropout(0.05)
        
        # 层归一化
        self.norm = nn.LayerNorm(node_dim)
    
    def process_edge_interactions(self, x, edge_index, edge_attr):
        """处理边特征与节点特征的交互 - 点乘方式"""
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        aggregated = torch.zeros_like(x)
        
        # 转换边特征：7维 → 8维
        transformed_edge_attr = self.edge_feature_transform(edge_attr)
        
        # 确保transformed_edge_attr的形状正确
        if transformed_edge_attr.dim() == 1:
            transformed_edge_attr = transformed_edge_attr.unsqueeze(0)
        
        for i in range(num_edges):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            
            if src >= num_nodes or dst >= num_nodes:
                continue
                
            # 获取源节点特征和转换后的边特征
            src_feature = x[src]  # [8]
            edge_feature = transformed_edge_attr[i]  # [8]
            
            # 使用点乘
            interaction_effect = src_feature * edge_feature  # [8] * [8] = [8]
            
            # 应用缩放参数
            scaled_interaction = interaction_effect * self.dot_product_scale
            
            # 聚合到目标节点
            aggregated[dst] += scaled_interaction
        
        return aggregated
    
    def forward(self, x, edge_index, edge_attr):
        """前向传播"""
        # 变换节点特征
        x_transformed = self.node_transform(x)
        
        edge_contributions = self.process_edge_interactions(x_transformed, edge_index, edge_attr)
        out = x + edge_contributions * self.residual_weight
        out = self.dropout(out)
        out = self.norm(out)
        return out


class FrameAwareNodeSelectionGNN(nn.Module):
    """帧感知节点选择GNN - 可配置突变效应层，改进版考虑节点度和类型"""
    def __init__(self, node_dim=8, edge_dim=7, num_layers=5, mutation_layers=2):
        super(FrameAwareNodeSelectionGNN, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.mutation_layers = mutation_layers  # 在哪些层引入突变效应
        
        # 突变特征变换器（与节点特征变换器独立）
        # 注意：这里我们有两种变换器：
        # 1. 从5维Atchley到node_dim的变换器（用于第一层）
        # 2. 从node_dim到node_dim的变换器（用于后续层）
        
        # 第一层突变变换器：将5维Atchley特征转换为node_dim
        self.first_mutation_transformer = nn.Sequential(
            nn.Linear(5, node_dim * 2),  # 从5维Atchley到高维
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(node_dim * 2, node_dim),  # 匹配节点特征维度
            nn.LayerNorm(node_dim)
        )
        
        # 后续层的突变变换器：从node_dim到node_dim
        self.layer_mutation_transformers = nn.ModuleList()
        for i in range(mutation_layers - 1):  # 减去第一层
            transformer = nn.Sequential(
                nn.Linear(node_dim, node_dim * 2),  # 从node_dim到高维
                nn.LeakyReLU(0.1),
                nn.Dropout(0.05),
                nn.Linear(node_dim * 2, node_dim),  # 保持node_dim
                nn.LayerNorm(node_dim)
            )
            self.layer_mutation_transformers.append(transformer)
        
        # 多层图卷积 - 前mutation_layers层使用Atchley变换
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            use_atchley = (i < mutation_layers)  # 只在指定层使用Atchley变换
            self.convs.append(EfficientGraphConv(node_dim, edge_dim, use_atchley_transform=use_atchley))
        
        # DiffPool层 - 改进版考虑节点度和类型
        self.diffpools = nn.ModuleList()
        for i in range(num_layers):
            self.diffpools.append(DiffPool(node_dim, edge_dim, 0.5))
        
        # 节点重要性评分器
        self.node_scorer = nn.Sequential(
            nn.Linear(10, 32),  # 节点特征(8) + 度中心性(1) + 节点类型(1)
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 1)
        )
        
        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(node_dim * 2, node_dim)
        )
        
        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.03),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 2)
        )
        
        # 残差连接权重（用于控制突变效应强度）
        self.mutation_residual_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(mutation_layers)
        ])
        
        self.print_counter = 0
        self.print_interval = 200
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def compute_node_importance(self, x, edge_index, node_types=None):
        """计算节点重要性分数"""
        num_nodes = x.size(0)
        
        if num_nodes == 0:
            return torch.zeros(0, device=x.device)
        
        # 计算度中心性
        device = x.device
        degree_centrality = torch.zeros(num_nodes, device=device)
        if edge_index.size(1) > 0:
            for i in range(edge_index.size(1)):
                src = edge_index[0, i]
                dst = edge_index[1, i]
                if src < num_nodes:
                    degree_centrality[src] += 1
                if dst < num_nodes:
                    degree_centrality[dst] += 1
        
        if degree_centrality.max() > 0:
            degree_centrality = degree_centrality / degree_centrality.max()
        
        # 处理节点类型
        if node_types is None:
            node_types = torch.zeros((num_nodes, 1), device=device)
        else:
            node_types = node_types.to(device)
            if node_types.dim() == 1:
                node_types = node_types.unsqueeze(1)
        
        # 拼接所有特征
        degree_centrality = degree_centrality.unsqueeze(1)
        combined_features = torch.cat([
            x,
            degree_centrality,
            node_types
        ], dim=1)
        
        # 计算重要性分数
        importance = self.node_scorer(combined_features).squeeze(1)
        return importance
    
    def forward(self, data, return_selection_info=False):
        """前向传播 - 改进版，在池化时考虑节点度和类型"""
        # 解包数据
        node_features = data['node_features']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
        mutant_indices = data.get('mutant_indices', [])
        mutant_atchley = data.get('mutant_atchley', torch.zeros((0, 5)))
        node_types = data.get('node_types', None)
        pdb_id = data.get('pdb_id', None)
        frame_idx = data.get('frame_idx', 0)
        
        # 获取设备
        device = next(self.parameters()).device
        
        # 将所有张量移动到正确的设备
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        mutant_atchley = mutant_atchley.to(device)
        
        # 确保node_types在正确的设备上
        if node_types is not None:
            if isinstance(node_types, torch.Tensor):
                if node_types.device != device:
                    node_types = node_types.to(device)
            else:
                # 如果node_types是列表或其他类型，转换为张量
                node_types = torch.tensor(node_types, device=device)
        else:
            # 如果node_types是None，创建零张量
            if node_features.dim() > 1:
                num_nodes = node_features.size(0)
            else:
                num_nodes = 1
            node_types = torch.zeros((num_nodes, 1), device=device)
        
        if node_features.dim() == 3:
            node_features = node_features.squeeze(0)
        
        x = node_features
        
        self.print_counter += 1
        should_print = self.print_counter % self.print_interval == 0
        
        # 存储选择信息
        frame_selection_info = {
            'pdb_id': pdb_id,
            'frame_idx': frame_idx,
            'layer_assignments': []
        }
        
        if should_print:
            print(f"\n=== 前向传播: PDB {pdb_id}, 帧 {frame_idx} ===")
            print(f"  突变节点: {mutant_indices}")
            print(f"  突变层数: {self.mutation_layers}/{self.num_layers}")
            if node_types is not None:
                # 确保node_types是二维的
                if node_types.dim() == 1:
                    node_types = node_types.unsqueeze(1)
                substrate_count = torch.sum(node_types == -1.0).item()
                wildtype_count = torch.sum(node_types == 0.0).item()
                mutant_count = torch.sum(node_types == 1.0).item()
                print(f"  初始节点类型统计: 底物={substrate_count}, 非突变残基={wildtype_count}, 突变残基={mutant_count}")
        
        # 1. 初始化突变效应矩阵
        mutation_effect_matrix = None
        if len(mutant_indices) > 0 and mutant_atchley.size(0) > 0 and self.mutation_layers > 0:
            # 为每个突变位点创建初始突变效应
            mutation_effects = []
            for i, idx in enumerate(mutant_indices):
                idx = int(idx)
                if idx < node_features.size(0):
                    # 使用第一层突变变换器处理突变特征（5维 -> 8维）
                    mutation_feat = self.first_mutation_transformer(mutant_atchley[i].unsqueeze(0))
                    mutation_effects.append((idx, mutation_feat.squeeze(0)))
            
            if mutation_effects:
                # 创建突变效应矩阵
                mutation_effect_matrix = torch.zeros_like(x)
                for idx, effect in mutation_effects:
                    mutation_effect_matrix[idx] = effect
        
        if should_print and mutation_effect_matrix is not None:
            print(f"  初始突变特征矩阵非零行数: {torch.sum(torch.norm(mutation_effect_matrix, dim=1) > 1e-6).item()}")
            print(f"  初始突变特征矩阵总强度: {torch.norm(mutation_effect_matrix).item():.4f}")
        
        # 2. 多层传播
        assignment_matrices = []
        layer_mutation_effects = []
        layer_node_types = []  # 保存每层的节点类型
        
        for layer_idx in range(self.num_layers):
            # 只在指定层引入突变效应
            if mutation_effect_matrix is not None and layer_idx < self.mutation_layers:
                current_mutation_effect = mutation_effect_matrix.clone()
                
                # 如果当前层有独立的突变变换器，重新变换
                # 注意：第一层（layer_idx=0）已经用first_mutation_transformer处理过了
                # 后续层（layer_idx>=1）使用layer_mutation_transformers
                if layer_idx > 0 and (layer_idx-1) < len(self.layer_mutation_transformers):
                    # 对突变效应矩阵应用当前层的变换
                    batch_size = current_mutation_effect.size(0)
                    flat_effect = current_mutation_effect.view(-1, self.node_dim)
                    transformed = self.layer_mutation_transformers[layer_idx-1](flat_effect)
                    current_mutation_effect = transformed.view(batch_size, -1)
                
                # 将突变效应加到节点特征上（带残差权重）
                alpha = torch.sigmoid(self.mutation_residual_weights[min(layer_idx, len(self.mutation_residual_weights)-1)])
                x_with_mutation = x + alpha * current_mutation_effect
            else:
                x_with_mutation = x
            
            if should_print and mutation_effect_matrix is not None and layer_idx < self.mutation_layers:
                if current_mutation_effect is not None:
                    mutation_contribution = torch.norm(current_mutation_effect).item()
                else:
                    mutation_contribution = 0.0
                x_norm = torch.norm(x).item()
                if x_norm > 0:
                    ratio = mutation_contribution / x_norm
                else:
                    ratio = 0.0
                print(f"  第{layer_idx+1}层突变贡献: {mutation_contribution:.4f}, 节点特征范数: {x_norm:.4f}, 比例: {ratio:.4f}")
            
            # 图卷积
            x = F.leaky_relu(self.convs[layer_idx](x_with_mutation, edge_index, edge_attr), 0.1)
            
            # 保存当前层节点类型用于池化
            current_node_types = node_types.clone()
            
            # 池化（现在包含节点类型）
            x, edge_index, edge_attr, assign, node_types = self.diffpools[layer_idx](
                x, edge_index, edge_attr, current_node_types
            )
            
            assignment_matrices.append(assign)
            layer_node_types.append(current_node_types)  # 保存节点类型
            
            # 池化突变效应矩阵（如果存在且在指定层）
            if mutation_effect_matrix is not None and layer_idx < self.mutation_layers:
                # 使用分配矩阵池化突变效应
                mutation_effect_matrix = torch.matmul(assign.t(), mutation_effect_matrix)
                layer_mutation_effects.append(mutation_effect_matrix.clone())
            else:
                layer_mutation_effects.append(None)
            
            if should_print:
                if node_types is not None:
                    # 确保node_types是二维的
                    if node_types.dim() == 1:
                        node_types = node_types.unsqueeze(1)
                    substrate_count = torch.sum(torch.abs(node_types + 1.0) < 0.1).item()  # 接近-1.0
                    mutant_count = torch.sum(torch.abs(node_types - 1.0) < 0.1).item()  # 接近1.0
                    wildtype_count = torch.sum(torch.abs(node_types) < 0.1).item()  # 接近0.0
                if mutation_effect_matrix is not None and layer_idx < self.mutation_layers:
                    print(f"    突变特征矩阵非零行数: {torch.sum(torch.norm(mutation_effect_matrix, dim=1) > 1e-6).item()}")
                    print(f"    突变特征矩阵总强度: {torch.norm(mutation_effect_matrix).item():.4f}")
        
        # 3. 特征变换
        x = self.feature_transform(x)
        
        # 4. 计算最终的重要性分数用于池化
        final_importance_scores = self.compute_node_importance(x, edge_index, node_types)
        
        # 使用重要性得分进行加权池化
        # 应用softmax归一化
        weights = F.softmax(final_importance_scores, dim=0)
        
        # 确保权重维度正确
        if weights.dim() == 1:
            weights = weights.unsqueeze(1)  # [num_nodes] -> [num_nodes, 1]
        
        # 加权池化
        x_pooled = torch.sum(x * weights, dim=0, keepdim=True)  # [1, node_dim]
        
        # 预测
        prediction = self.predictor(x_pooled)
        
        if return_selection_info:
            frame_selection_info['layer_assignments'] = assignment_matrices
            frame_selection_info['layer_mutation_effects'] = layer_mutation_effects
            frame_selection_info['layer_node_types'] = layer_node_types
            frame_selection_info['final_weights'] = weights.squeeze(1)  # 保存最终的权重
            return prediction.squeeze(0), frame_selection_info
        
        return prediction.squeeze(0)

# 其他函数保持不变...

def load_targets(file_path):
    """加载目标值文件"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(1024)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    print(f"检测到目标值文件编码: {encoding}")
    
    # 使用检测到的编码读取文件
    targets_df = pd.read_csv(file_path, header=None, encoding=encoding)
    
    # 提取有效的目标值
    valid_targets = []
    for i in range(len(targets_df)):
        row = targets_df.iloc[i]
        if len(row) >= 2:
            try:
                val1 = float(row[0])
                val2 = float(row[1])
                valid_targets.append([val1, val2])
            except (ValueError, TypeError):
                print(f"警告: 忽略无效行 {i}: 前两列值 {row[0]}, {row[1]}")
                valid_targets.append([0.5, 0.5])
        else:
            print(f"警告: 行 {i} 列数不足: {len(row)}")
            valid_targets.append([0.5, 0.5])
    
    return np.array(valid_targets)

def load_data(data_dir):
    """加载所有数据"""
    # 加载关键帧索引
    key_frames = np.load("key_frames.npy")
    print(f"加载关键帧文件, 形状: {key_frames.shape}")
    
    # 加载目标值
    targets = load_targets("targets.csv")
    print(f"加载目标值文件, 形状: {targets.shape}")
    
    # 确保关键帧和目标值数量匹配
    if len(key_frames) != len(targets):
        min_len = min(len(key_frames), len(targets))
        key_frames = key_frames[:min_len]
        targets = targets[:min_len]
        print(f"调整关键帧和目标值数量为 {min_len} 以匹配")
    
    # 加载Atchley特征
    atchley_features = []
    for i in range(1, len(targets) + 1):
        atchley_file = os.path.join(data_dir, "atchley", f"{i}_atchley.npy")
        if os.path.exists(atchley_file):
            atchley_feat = np.load(atchley_file)
            atchley_features.append(atchley_feat)
        else:
            print(f"警告: Atchley文件不存在 {atchley_file}, 使用零向量")
            atchley_features.append(np.zeros(5))
    
    atchley_features = np.array(atchley_features)
    print(f"加载Atchley特征, 形状: {atchley_features.shape}")
    
    # 加载图数据
    graph_data = []
    for i in range(1, len(targets) + 1):
        graph_file = os.path.join(data_dir, "graph", f"{i}_graph.npy")
        if os.path.exists(graph_file):
            graph_feat = np.load(graph_file, allow_pickle=True)
            graph_data.append(graph_feat)
        else:
            print(f"警告: 图数据文件不存在 {graph_file}, 使用随机数据")
            graph_feat = np.array([{
                'node_features': np.random.randn(1, 8),
                'edge_index': np.zeros((2, 0), dtype=np.int64),
                'edge_attr': np.zeros((0, 7), dtype=np.float32),
                'mutant_nodes': [{
                    'index': 0,
                    'atchley_feature': np.zeros(5)
                }]
            } for _ in range(10)])
            graph_data.append(graph_feat)
    
    return key_frames, targets, atchley_features, graph_data

def prepare_data_by_frame(graph_data, atchley_features, targets, train_ratio=0.6, val_ratio=0.2):
    """按帧划分数据，每个PDB的帧随机分配到不同数据集"""
    train_data = []
    train_targets = []
    val_data = []
    val_targets = []
    test_data = []
    test_targets = []
    
    # 收集所有帧样本
    all_frames = []
    all_frame_targets = []
    all_frame_info = []
    
    # 获取所有PDB的数量
    num_pdbs = len(graph_data)
    print(f"📊 总PDB数量: {num_pdbs}")
    
    # 遍历所有PDB，收集所有帧
    total_frames = 0
    for pdb_id in range(num_pdbs):
        # 每个样本的图数据是一个数组，包含多个关键帧的数据
        sample_graph_data = graph_data[pdb_id]
        num_frames = len(sample_graph_data)
        total_frames += num_frames
        
        # 从Atchley特征中识别突变位点
        atchley_feat = atchley_features[pdb_id]
        mutant_indices = []
        
        # 确保atchley_feat是二维的
        if atchley_feat.ndim == 1:
            atchley_feat = atchley_feat.reshape(-1, 5)
        
        # 识别非零向量作为突变位点
        for residue_idx in range(len(atchley_feat)):
            vector = atchley_feat[residue_idx]
            if np.any(vector != 0):
                mutant_indices.append(residue_idx)
        
        print(f"PDB {pdb_id+1} 的突变位点: {mutant_indices} (从Atchley特征识别), 帧数: {num_frames}")
        
        # 处理每个关键帧
        for frame_idx, frame in enumerate(sample_graph_data):
            # 提取节点特征
            if 'node_features' in frame and frame['node_features'] is not None:
                node_features = frame['node_features']
            else:
                print(f"警告: PDB {pdb_id+1} 的关键帧 {frame_idx} 缺少 'node_features', 使用随机数据")
                node_features = np.random.randn(1, 8)
            
            if node_features.ndim == 1:
                node_features = node_features.reshape(1, -1)
            
            # 提取边索引和边特征
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 7), dtype=np.float32)
            
            if 'edges' in frame and frame['edges'] is not None:
                num_edges = len(frame['edges'])
                edge_index_arr = np.zeros((2, num_edges), dtype=np.int64)
                edge_attr_arr = np.zeros((num_edges, 7), dtype=np.float32)
                
                for idx, edge in enumerate(frame['edges']):
                    edge_index_arr[0, idx] = edge['source']
                    edge_index_arr[1, idx] = edge['target']
                    edge_attr_arr[idx] = edge['features']
                
                edge_index = edge_index_arr
                edge_attr = edge_attr_arr
            elif 'edge_index' in frame and frame['edge_index'] is not None:
                edge_index = frame['edge_index']
                edge_attr = frame['edge_attr']
            
            # 准备突变节点信息
            mutant_atchley_list = []
            valid_mutant_indices = []
            
            for mut_idx in mutant_indices:
                if mut_idx < node_features.shape[0]:
                    valid_mutant_indices.append(mut_idx)
                    mutant_atchley_list.append(atchley_feat[mut_idx])
                else:
                    print(f"警告: 突变位点索引 {mut_idx} 超出节点范围 (0-{node_features.shape[0]-1})")
            
            # 转换为numpy数组
            if mutant_atchley_list:
                mutant_atchley_array = np.array(mutant_atchley_list)
            else:
                mutant_atchley_array = np.zeros((0, 5))
            
            # 计算节点类型特征：底物(-1)、非突变残基(0)、突变残基(1)
            num_nodes = node_features.shape[0]
            node_types = np.zeros((num_nodes, 1), dtype=np.float32)  # 二维数组：[num_nodes, 1]
            
            # 首先将所有节点标记为残基(0)
            node_types.fill(0.0)
            
            # 标记突变残基(1)
            for mut_idx in valid_mutant_indices:
                if mut_idx < num_nodes:
                    node_types[mut_idx, 0] = 1.0  # 突变残基标记为1
            
            # 标记底物(-1) - 通过检查节点特征是否全为0
            for i in range(num_nodes):
                node_feat = node_features[i]
                # 检查节点特征是否全为0
                if np.all(np.abs(node_feat) < 1e-8):  # 使用小的阈值
                    node_types[i, 0] = -1.0  # 底物标记为-1
                    # 但如果是突变节点，应该保持为1
                    if i in valid_mutant_indices:
                        print(f"警告: 节点{i}既是底物又是突变节点，这可能不合理")
            
            # 统计节点类型分布
            substrate_count = np.sum(node_types == -1.0)
            wildtype_count = np.sum(node_types == 0.0)
            mutant_count = np.sum(node_types == 1.0)
            
            if frame_idx == 0:  # 只打印第一帧的统计信息
                print(f"  PDB {pdb_id+1} 节点类型统计:")
                print(f"    底物节点: {substrate_count}")
                print(f"    非突变残基: {wildtype_count}")
                print(f"    突变残基: {mutant_count}")
                print(f"    总节点数: {num_nodes}")
            
            # 准备数据样本
            sample = {
                'pdb_id': pdb_id,
                'frame_idx': frame_idx,
                'node_features': torch.tensor(node_features, dtype=torch.float32),
                'edge_index': torch.tensor(edge_index, dtype=torch.long),
                'edge_attr': torch.tensor(edge_attr, dtype=torch.float32),
                'atchley_feature': torch.tensor(atchley_features[pdb_id], dtype=torch.float32),
                'mutant_indices': valid_mutant_indices,
                'mutant_atchley': torch.tensor(mutant_atchley_array, dtype=torch.float32),
                'node_types': torch.tensor(node_types, dtype=torch.float32)  # 添加节点类型
            }
            
            target = torch.tensor(targets[pdb_id], dtype=torch.float32)
            
            # 收集所有帧
            all_frames.append(sample)
            all_frame_targets.append(target)
            all_frame_info.append({
                'pdb_id': pdb_id,
                'frame_idx': frame_idx,
                'num_frames': num_frames
            })
    
    print(f"📊 总帧数: {total_frames}")
    
    # 随机打乱所有帧
    indices = list(range(len(all_frames)))
    random.shuffle(indices)
    
    # 计算划分点
    train_end = int(len(all_frames) * train_ratio)
    val_end = train_end + int(len(all_frames) * val_ratio)
    
    # 划分帧索引
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # 根据索引分配数据
    for idx in train_indices:
        train_data.append(all_frames[idx])
        train_targets.append(all_frame_targets[idx])
    
    for idx in val_indices:
        val_data.append(all_frames[idx])
        val_targets.append(all_frame_targets[idx])
    
    for idx in test_indices:
        test_data.append(all_frames[idx])
        test_targets.append(all_frame_targets[idx])
    
    # 统计每个PDB在不同数据集中的帧分布
    train_pdb_dist = {}
    val_pdb_dist = {}
    test_pdb_dist = {}
    
    for idx in train_indices:
        pdb_id = all_frame_info[idx]['pdb_id']
        train_pdb_dist[pdb_id] = train_pdb_dist.get(pdb_id, 0) + 1
    
    for idx in val_indices:
        pdb_id = all_frame_info[idx]['pdb_id']
        val_pdb_dist[pdb_id] = val_pdb_dist.get(pdb_id, 0) + 1
    
    for idx in test_indices:
        pdb_id = all_frame_info[idx]['pdb_id']
        test_pdb_dist[pdb_id] = test_pdb_dist.get(pdb_id, 0) + 1
    
    print(f"📊 按帧划分完成:")
    print(f"  训练集: {len(train_data)}帧, 涉及 {len(train_pdb_dist)}个PDB")
    print(f"  验证集: {len(val_data)}帧, 涉及 {len(val_pdb_dist)}个PDB")
    print(f"  测试集: {len(test_data)}帧, 涉及 {len(test_pdb_dist)}个PDB")
    
    # 打印PDB分布详情
    print(f"📊 PDB分布详情:")
    for pdb_id in range(num_pdbs):
        total_frames_pdb = sum(1 for info in all_frame_info if info['pdb_id'] == pdb_id)
        train_frames = train_pdb_dist.get(pdb_id, 0)
        val_frames = val_pdb_dist.get(pdb_id, 0)
        test_frames = test_pdb_dist.get(pdb_id, 0)
        print(f"  PDB {pdb_id+1}: 总帧数={total_frames_pdb}, 训练={train_frames}, 验证={val_frames}, 测试={test_frames}")
    
    return train_data, train_targets, val_data, val_targets, test_data, test_targets

class ProteinDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.indices = list(range(len(data)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.data[actual_idx].copy()
        sample['target'] = self.targets[actual_idx]
        return sample
    
    def shuffle(self):
        random.shuffle(self.indices)

class AdaptiveStdWeightedMSELoss(nn.Module):
    """自适应标准差加权MSE损失函数 - 结合标准差加权和分段线性特性"""
    def __init__(self, train_targets=None, threshold=5.0, low_slope=0.1, 
                 high_slope=2.0, reduction='mean'):
        super(AdaptiveStdWeightedMSELoss, self).__init__()
        self.reduction = reduction
        self.threshold = threshold      # 分段阈值
        self.low_slope = low_slope      # 小误差区域的斜率
        self.high_slope = high_slope    # 大误差区域的斜率
        
        # 标准差权重和分段斜率权重
        self.std_weights = None
        self.slope_weights = None
        
        # 如果提供了训练数据，自动计算基于标准差的权重
        if train_targets is not None:
            self.compute_std_weights_from_data(train_targets)
    
    def compute_std_weights_from_data(self, train_targets):
        """根据训练数据的标准差计算权重"""
        if len(train_targets) == 0:
            # 如果没有数据，使用默认权重
            self.std_weights = torch.tensor([0.5, 0.5])
            return
        
        # 将目标值转换为numpy数组
        if isinstance(train_targets[0], torch.Tensor):
            targets_array = torch.stack(train_targets).cpu().numpy()
        else:
            targets_array = np.array(train_targets)
        
        # 计算每个维度的标准差
        std_1_3 = np.std(targets_array[:, 0])  # 1-3键比例的标准差
        std_1_6 = np.std(targets_array[:, 1])  # 1-6键比例的标准差

        print(f"  1-3键比例标准差: {std_1_3:.4f}")
        print(f"  1-6键比例标准差: {std_1_6:.4f}")
        
        # 基于标准差计算权重（标准差越大，权重越小）
        weight_1_3 = 1.0 / std_1_3
        weight_1_6 = 1.0 / std_1_6
        
        # 归一化权重，使它们之和为1
        total_weight = weight_1_3 + weight_1_6
        weight_1_3_normalized = weight_1_3 / total_weight
        weight_1_6_normalized = weight_1_6 / total_weight
        
        self.std_weights = torch.tensor([weight_1_3_normalized, weight_1_6_normalized])
        
        # 打印权重信息
        print(f"  标准差权重: 1-3键={weight_1_3_normalized:.4f}, 1-6键={weight_1_6_normalized:.4f}")
        print(f"  分段斜率: 小误差({self.low_slope}) | 大误差({self.high_slope}), 阈值={self.threshold}")
    
    def compute_slope_based_weights(self, errors):
        """根据误差大小计算分段斜率权重"""
        # 创建与误差相同形状的斜率权重
        slope_weights = torch.ones_like(errors)
        
        # 小误差区域：应用低斜率
        small_error_mask = errors <= self.threshold
        slope_weights[small_error_mask] = self.low_slope
        
        # 大误差区域：应用高斜率
        large_error_mask = errors > self.threshold
        slope_weights[large_error_mask] = self.high_slope
        
        return slope_weights
    
    def forward(self, input, target):
        # 如果没有计算标准差权重，使用默认等权重
        if self.std_weights is None:
            self.std_weights = torch.tensor([0.5, 0.5])
        
        # 确保权重在正确的设备上
        if self.std_weights.device != input.device:
            self.std_weights = self.std_weights.to(input.device)
        
        # 计算误差
        errors = torch.abs(input - target)
        squared_errors = (input - target) ** 2
        
        # 计算分段斜率权重
        slope_weights = self.compute_slope_based_weights(errors)
        
        # 结合标准差权重和分段斜率权重
        # std_weights形状: [1, 2] (从[2]扩展)
        std_weights_expanded = self.std_weights.unsqueeze(0)  # 从 [2] 变为 [1, 2]
        
        # 计算最终的加权损失
        # 公式: loss = std_weight * slope_weight * squared_error
        weighted_mse = std_weights_expanded * slope_weights * squared_errors
        
        if self.reduction == 'mean':
            return torch.mean(weighted_mse)
        elif self.reduction == 'sum':
            return torch.sum(weighted_mse)
        else:
            return weighted_mse


def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs, patience=5, scheduler=None):
    """训练函数"""
    device = next(model.parameters()).device
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        print(f"\n🎯 Epoch {epoch+1}/{epochs}")
        
        # 训练阶段
        model.train()
        train_dataloader.dataset.shuffle()
        epoch_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch_loss = 0.0
            for sample in batch:
                inputs = {
                    'node_features': sample['node_features'].to(device),
                    'edge_index': sample['edge_index'].to(device),
                    'edge_attr': sample['edge_attr'].to(device),
                    'atchley_feature': sample['atchley_feature'].to(device),
                    'mutant_indices': sample['mutant_indices'],
                    'mutant_atchley': sample['mutant_atchley'].to(device),
                    'node_types': sample.get('node_types', None),
                    'pdb_id': sample.get('pdb_id', None),
                    'frame_idx': sample.get('frame_idx', 0)
                }
                target = sample['target'].to(device)
                
                # 前向传播
                prediction = model(inputs, return_selection_info=False)
                loss = criterion(prediction, target)
                batch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_train_loss += batch_loss / len(batch)
            if (batch_idx + 1) % 2 == 0:
                print(f"  批次 {batch_idx+1}/{len(train_dataloader)}, 平均损失: {epoch_train_loss/(batch_idx+1):.6f}")
        
        epoch_train_loss /= len(train_dataloader)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch_loss = 0.0
                for sample in batch:
                    inputs = {
                        'node_features': sample['node_features'].to(device),
                        'edge_index': sample['edge_index'].to(device),
                        'edge_attr': sample['edge_attr'].to(device),
                        'atchley_feature': sample['atchley_feature'].to(device),
                        'mutant_indices': sample['mutant_indices'],
                        'mutant_atchley': sample['mutant_atchley'].to(device),
                        'node_types': sample.get('node_types', None),
                        'pdb_id': sample.get('pdb_id', None),
                        'frame_idx': sample.get('frame_idx', 0)
                    }
                    target = sample['target'].to(device)
                    
                    # 前向传播
                    prediction = model(inputs, return_selection_info=False)
                    loss = criterion(prediction, target)
                    batch_loss += loss.item()
                
                epoch_val_loss += batch_loss / len(batch)
        
        epoch_val_loss /= len(val_dataloader)
        
        # 学习率调整
        if scheduler:
            scheduler.step(epoch_val_loss)
        
        print(f"📊 Epoch {epoch+1} 结果:")
        print(f"  训练损失: {epoch_train_loss:.6f}")
        print(f"  验证损失: {epoch_val_loss:.6f}")
        
        # 早停检查
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ 验证损失改善, 保存最佳模型: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️  验证损失未改善, 连续 {epochs_no_improve}/{patience} 个epoch")
            
            if epochs_no_improve >= patience:
                print("🛑 早停触发!")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load("best_model.pth"))
    return best_val_loss

def evaluate(model, dataloader, criterion):
    """评估模型"""
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_loss = 0.0
            for sample in batch:
                inputs = {
                    'node_features': sample['node_features'].to(device),
                    'edge_index': sample['edge_index'].to(device),
                    'edge_attr': sample['edge_attr'].to(device),
                    'atchley_feature': sample['atchley_feature'].to(device),
                    'mutant_indices': sample['mutant_indices'],
                    'mutant_atchley': sample['mutant_atchley'].to(device),
                    'node_types': sample.get('node_types', None),
                    'pdb_id': sample.get('pdb_id', None),
                    'frame_idx': sample.get('frame_idx', 0)
                }
                target = sample['target'].to(device)
                
                # 前向传播
                prediction = model(inputs, return_selection_info=False)
                loss = criterion(prediction, target)
                batch_loss += loss.item()
                
                all_preds.append(prediction.detach().cpu().numpy())
                all_targets.append(target.detach().cpu().numpy())
            
            total_loss += batch_loss / len(batch)
    
    # 计算R2分数
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    r2 = 1 - np.sum((all_targets - all_preds)**2) / np.sum((all_targets - np.mean(all_targets))**2)
    
    return total_loss / len(dataloader), r2

def predict(model, dataloader):
    """预测函数"""
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_info = []
    all_selection_info = []
    
    with torch.no_grad():
        for batch in dataloader:
            for sample in batch:
                inputs = {
                    'node_features': sample['node_features'].to(device),
                    'edge_index': sample['edge_index'].to(device),
                    'edge_attr': sample['edge_attr'].to(device),
                    'atchley_feature': sample['atchley_feature'].to(device),
                    'mutant_indices': sample['mutant_indices'],
                    'mutant_atchley': sample['mutant_atchley'].to(device),
                    'node_types': sample.get('node_types', None),
                    'pdb_id': sample.get('pdb_id', None),
                    'frame_idx': sample.get('frame_idx', 0)
                }
                
                # 前向传播
                prediction, selection_info = model(inputs, return_selection_info=True)
                
                all_preds.append(prediction.detach().cpu().numpy())
                all_info.append({
                    'pdb_id': sample.get('pdb_id', None),
                    'frame_idx': sample.get('frame_idx', 0)
                })
                all_selection_info.append(selection_info)
    
    return np.array(all_preds), all_info, all_selection_info

# [其他类和函数保持不变...]

def main():
    data_dir = "data"
    learning_rate = 0.001
    epochs = 1000
    patience = 9
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    key_frames, targets, atchley_features, graph_data = load_data(data_dir)

    # 使用按帧划分
    train_data, train_targets, val_data, val_targets, test_data, test_targets = prepare_data_by_frame(
        graph_data, atchley_features, targets
    )
    
    # 创建数据集
    train_dataset = ProteinDataset(train_data, train_targets)
    val_dataset = ProteinDataset(val_data, val_targets)
    test_dataset = ProteinDataset(test_data, test_targets)
    
    # 数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: x)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x)

    model = FrameAwareNodeSelectionGNN(
        node_dim=8,
        edge_dim=7,
        num_layers=5,
    ).to(device)
    
    # 优化器和损失函数
    criterion = AdaptiveStdWeightedMSELoss(train_targets=train_targets, reduction='mean').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2, verbose=True)
    
    # 训练模型
    best_val_loss = train_model(
        model, train_dataloader, val_dataloader, optimizer, criterion, epochs, patience, scheduler
    )
    
    # 评估模型
    train_loss, train_r2 = evaluate(model, train_dataloader, criterion)
    val_loss, val_r2 = evaluate(model, val_dataloader, criterion)
    
    print(f"训练集 Loss: {train_loss:.6f}, R2: {train_r2:.4f}")
    print(f"验证集 Loss: {val_loss:.6f}, R2: {val_r2:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "node_selection_gnn_model.pth")

    # 测试集预测
    test_preds, test_info, test_selection_info = predict(model, test_dataloader)

    # 保存测试结果
    test_results = []
    test_pdb_ids = set(info['pdb_id'] for info in test_info)

    for pdb_id in test_pdb_ids:
        # 获取该PDB的所有预测
        pdb_preds = [pred for pred, info in zip(test_preds, test_info) if info['pdb_id'] == pdb_id]
        pdb_frames = [info for info in test_info if info['pdb_id'] == pdb_id]
        
        # 计算平均预测
        avg_pred = np.mean(pdb_preds, axis=0)
        
        # 获取该PDB的真实目标值
        pdb_target = targets[pdb_id]
        target_1_3, target_1_6 = pdb_target
        pred_1_3, pred_1_6 = avg_pred
        
        # 计算误差
        error_1_3 = abs(target_1_3 - pred_1_3)
        error_1_6 = abs(target_1_6 - pred_1_6)
        total_error = error_1_3 + error_1_6
        
        test_results.append({
            'pdb_id': pdb_id,
            'num_frames': len(pdb_preds),
            'target_1_3': target_1_3,
            'target_1_6': target_1_6,
            'pred_1_3': pred_1_3,
            'pred_1_6': pred_1_6,
            'error_1_3': error_1_3,
            'error_1_6': error_1_6,
            'total_error': total_error,
            'avg_pred_1_3': np.mean([p[0] for p in pdb_preds]),
            'avg_pred_1_6': np.mean([p[1] for p in pdb_preds]),
            'std_pred_1_3': np.std([p[0] for p in pdb_preds]) if len(pdb_preds) > 1 else 0,
            'std_pred_1_6': np.std([p[1] for p in pdb_preds]) if len(pdb_preds) > 1 else 0,
            'pred_range_1_3': np.ptp([p[0] for p in pdb_preds]) if len(pdb_preds) > 1 else 0,
            'pred_range_1_6': np.ptp([p[1] for p in pdb_preds]) if len(pdb_preds) > 1 else 0
        })

    # 保存测试结果
    df_test_results = pd.DataFrame(test_results)
    df_test_results.to_csv("test_results.csv", index=False)
    
    print(f"✅ 测试结果已保存: test_results.csv ({len(df_test_results)} 个PDB)")
    
    # 打印总体统计
    print(f"\n📊 测试集总体统计:")
    print(f"  平均1-3键误差: {df_test_results['error_1_3'].mean():.4f} ± {df_test_results['error_1_3'].std():.4f}")
    print(f"  平均1-6键误差: {df_test_results['error_1_6'].mean():.4f} ± {df_test_results['error_1_6'].std():.4f}")
    print(f"  平均总误差: {df_test_results['total_error'].mean():.4f} ± {df_test_results['total_error'].std():.4f}")
    print(f"  预测稳定性 - 1-3键标准差: {df_test_results['std_pred_1_3'].mean():.4f}")
    print(f"  预测稳定性 - 1-6键标准差: {df_test_results['std_pred_1_6'].mean():.4f}")

    # 保存模型性能总结
    performance_summary = {
        'train_loss': train_loss,
        'train_r2': train_r2,
        'val_loss': val_loss,
        'val_r2': val_r2,
        'test_avg_error_1_3': df_test_results['error_1_3'].mean(),
        'test_avg_error_1_6': df_test_results['error_1_6'].mean(),
        'test_avg_total_error': df_test_results['total_error'].mean(),
        'test_pred_stability_1_3': df_test_results['std_pred_1_3'].mean(),
        'test_pred_stability_1_6': df_test_results['std_pred_1_6'].mean(),
        'num_test_pdbs': len(df_test_results)
    }

    df_summary = pd.DataFrame([performance_summary])
    df_summary.to_csv("model_performance_summary.csv", index=False)
    print(f"  - 模型性能总结: model_performance_summary.csv")
        
if __name__ == "__main__":
    main()