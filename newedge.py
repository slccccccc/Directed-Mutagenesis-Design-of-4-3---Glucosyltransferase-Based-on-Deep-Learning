import numpy as np
from scipy.spatial.distance import cdist
import os
import re
import glob

# 氨基酸与保留原子的映射
RESIDUE_ATOM_MAP = {
    'ASP': 'OD2', 'GLU': 'OE2', 'LYS': 'NZ', 'ARG': 'CZ', 
    'ASN': 'ND2', 'GLN': 'NE2', 'SER': 'OG', 'THR': 'OG1',
    'TYR': 'OH', 'HIE': 'ND1', 'CYS': 'SG', 'PHE': 'CZ',
    'TRP': 'NE1', 'ALA': 'CB', 'VAL': 'CB', 'LEU': 'CG',
    'ILE': 'CB', 'MET': 'SD', 'PRO': 'CG', 'GLY': 'CA'
}

# 疏水残基列表
HYDROPHOBIC_RESIDUES = {'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'MET', 'TRP'}

# 氢键供体残基
HBOND_DONORS = {'SER', 'THR', 'TYR', 'ASN', 'GLN', 'HIE', 'TRP', 'LYS', 'ARG'}

# 氢键受体残基
HBOND_ACCEPTORS = {'ASP', 'GLU', 'ASN', 'GLN', 'HIE', 'TYR', 'SER', 'THR'}

# 带正电残基 (阳离子)
POSITIVE_RESIDUES = {'LYS', 'ARG'}

# 带负电残基
NEGATIVE_RESIDUES = {'ASP', 'GLU'}

# 芳香族残基 (π系统)
AROMATIC_RESIDUES = {'PHE', 'TYR', 'TRP'}

def parse_pdb_file(pdb_path):
    """解析PDB文件，返回所有帧的原子信息"""
    frames = []
    current_frame = []
    in_model = False
    
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("MODEL"):
                    in_model = True
                    current_frame = []
                elif line.startswith("ENDMDL"):
                    in_model = False
                    if current_frame:
                        frames.append(current_frame)
                elif line.startswith("ATOM"):
                    if not in_model:
                        if not frames:
                            frames.append([])
                        frames[0].append(line)
                    else:
                        current_frame.append(line)
        
        if not frames and current_frame:
            frames.append(current_frame)
        
        return frames
    except Exception as e:
        print(f"解析PDB文件错误: {e}")
        return []

def extract_atom_info(line):
    """从PDB行中提取原子信息"""
    try:
        # 检查行长度是否足够
        if len(line) < 54:
            return None
            
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        
        # 检查残基名称是否有效
        if len(residue_name) != 3 or not residue_name.isalpha():
            return None
            
        residue_number = int(line[22:26].strip())
        chain_id = line[21] if line[21] != ' ' else 'A'
        
        # 检查坐标字段是否有效
        x_str = line[30:38].strip()
        y_str = line[38:46].strip()
        z_str = line[46:54].strip()
        
        if not x_str or not y_str or not z_str:
            return None
            
        x = float(x_str)
        y = float(y_str)
        z = float(z_str)
        
        return {
            'atom_name': atom_name,
            'residue_name': residue_name,
            'residue_number': residue_number,
            'chain_id': chain_id,
            'coords': np.array([x, y, z])
        }
    except (ValueError, IndexError):
        return None
def extract_graph_data(frame_lines, close_residues):
    """
    从PDB帧中提取图结构数据，处理所有close_residues中的残基和所有底物原子
    """
    # 存储残基和底物信息
    residue_info = {}
    protein_coords = []
    substrate_coords = []
    substrate_info = {}  # 存储底物原子信息
    
    # 映射残基编号到索引
    residue_idx_map = {}
    current_idx = 0
    substrate_idx = 0
    
    for line in frame_lines:
        atom_data = extract_atom_info(line)
        if not atom_data:
            continue
            
        resname = atom_data['residue_name']
        resnum = atom_data['residue_number']
        atom_name = atom_data['atom_name']
        coords = atom_data['coords']
        
        # 处理蛋白质残基 - 只考虑close_residues中的残基
        if resnum in close_residues:
            # 检查是否是代表原子
            if atom_name == RESIDUE_ATOM_MAP.get(resname, 'CA'):
                if resnum not in residue_idx_map:
                    residue_idx_map[resnum] = current_idx
                    residue_info[current_idx] = {
                        'resname': resname,
                        'residue_number': resnum,
                        'atom_name': atom_name,
                        'coords': coords
                    }
                    protein_coords.append(coords)
                    current_idx += 1
        
        # 处理底物 (UNL)
        elif resname == 'UNL':
            substrate_coords.append(coords)
            substrate_info[substrate_idx] = {
                'atom_name': atom_name,
                'coords': coords
            }
            substrate_idx += 1
    
    # 转换为numpy数组
    protein_coords = np.array(protein_coords) if protein_coords else np.zeros((0, 3))
    substrate_coords = np.array(substrate_coords) if substrate_coords else np.zeros((0, 3))
    
    return {
        'residue_info': residue_info,
        'protein_coords': protein_coords,
        'substrate_coords': substrate_coords,
        'substrate_info': substrate_info,
        'residue_idx_map': residue_idx_map,
        'num_residues': len(protein_coords),
        'num_substrate': len(substrate_coords)
    }

def calculate_edge_features(frame_data):
    """
    计算边特征 - 简化版本，只要距离小于5埃就成边并计算所有可能的相互作用
    底物原子作为单独节点处理
    """
    protein_coords = frame_data['protein_coords']
    substrate_coords = frame_data['substrate_coords']
    residue_info = frame_data['residue_info']
    substrate_info = frame_data['substrate_info']
    
    edges = []
    
    # 1. 残基-残基边
    n_res = len(protein_coords)
    if n_res > 1:
        # 计算残基间距离矩阵
        dist_matrix = cdist(protein_coords, protein_coords)
        
        for i in range(n_res):
            for j in range(i+1, n_res):
                r = dist_matrix[i, j]
                if r > 9.0:  # 距离阈值
                    continue
                    
                res_i = residue_info[i]
                res_j = residue_info[j]
                
                # 计算残基编号差
                res_diff = abs(res_i['residue_number'] - res_j['residue_number'])
                
                # 初始化特征向量
                features = np.zeros(7)
                
                # 修改点: 将第一维特征改为共价键指示器
                # 如果残基编号差为1，表示相邻残基（可能有共价键）
                features[0] = 1.0 if res_diff == 1 else 0.0
                
                # 1. 疏水相互作用
                if (res_i['resname'] in HYDROPHOBIC_RESIDUES and 
                    res_j['resname'] in HYDROPHOBIC_RESIDUES):
                    features[1] = 16.0 / (r**2)
                
                # 2. 氢键
                if ((res_i['resname'] in HBOND_DONORS and res_j['resname'] in HBOND_ACCEPTORS) or
                    (res_i['resname'] in HBOND_ACCEPTORS and res_j['resname'] in HBOND_DONORS)):
                    features[2] = 324.0 / (r**4)
                
                # 3. 盐桥
                if ((res_i['resname'] in POSITIVE_RESIDUES and res_j['resname'] in NEGATIVE_RESIDUES) or
                    (res_i['resname'] in NEGATIVE_RESIDUES and res_j['resname'] in POSITIVE_RESIDUES)):
                    features[3] = 18.0 / r
                
                # 4. π-π堆积
                if (res_i['resname'] in AROMATIC_RESIDUES and 
                    res_j['resname'] in AROMATIC_RESIDUES):
                    features[4] = 5442.0 / (r**6)
                
                # 5. π-阳离子相互作用
                if ((res_i['resname'] in AROMATIC_RESIDUES and res_j['resname'] in POSITIVE_RESIDUES) or
                    (res_i['resname'] in POSITIVE_RESIDUES and res_j['resname'] in AROMATIC_RESIDUES)):
                    features[5] = 33.0 / (r**2)
                
                # 6. 通用相互作用强度
                features[6] = 9.0 / r
                
                edges.append({
                    'type': 'res-res',
                    'source': i,
                    'target': j,
                    'distance': r,
                    'features': features
                })
    
    # 2. 残基-底物边 (每个底物原子单独处理)
    n_sub = len(substrate_coords)
    if n_sub > 0 and n_res > 0:
        # 计算残基到底物的距离
        res_sub_dist = cdist(protein_coords, substrate_coords)
        
        for i in range(n_res):
            for j in range(n_sub):
                r = res_sub_dist[i, j]
                if r > 5.0:  # 距离阈值
                    continue
                    
                res_i = residue_info[i]
                
                # 初始化特征向量
                features = np.zeros(7)
                features[0] = -1  # 残基-底物边赋-1
                
                # 1. 疏水相互作用
                if res_i['resname'] in HYDROPHOBIC_RESIDUES:
                    features[1] = 16.0 / (r**2)
                
                # 2. 氢键
                if (res_i['resname'] in HBOND_DONORS or res_i['resname'] in HBOND_ACCEPTORS):
                    features[2] = 324.0 / (r**4)
                
                # 3. 盐桥
                if res_i['resname'] in POSITIVE_RESIDUES or res_i['resname'] in NEGATIVE_RESIDUES:
                    features[3] = 18.0 / r
                
                features[4] = 0
                features[5] = 0
                
                # 6. 通用相互作用强度
                features[6] = 9.0 / r
                
                # 底物节点索引（在残基之后）
                substrate_node_idx = n_res + j
                
                edges.append({
                    'type': 'res-sub',
                    'source': i,
                    'target': substrate_node_idx,  # 指向特定的底物原子节点
                    'distance': r,
                    'features': features
                })
    
    return edges

def process_key_frames_for_pdb(pdb_path, key_frame_indices, close_residues, residue_features_dir):
    """
    处理单个PDB文件的关键帧，并输出每帧的边数和节点特征
    """
    # 解析PDB文件
    all_frames = parse_pdb_file(pdb_path)
    if not all_frames:
        print(f"无法解析PDB文件: {pdb_path}")
        return []
    
    # 获取PDB ID（文件名中的数字部分）
    pdb_id = os.path.basename(pdb_path).split('.')[0]
    
    # 加载残基特征文件
    residue_features_file = os.path.join(residue_features_dir, f"{pdb_id}.npy")
    if not os.path.exists(residue_features_file):
        print(f"警告: 残基特征文件不存在 {residue_features_file}")
        return []
    
    residue_features_all = np.load(residue_features_file)
    print(f"加载残基特征文件: {residue_features_file}, 形状: {residue_features_all.shape}")
    
    # 存储每个关键帧的结果
    key_frame_results = []
    
    # 处理每个关键帧
    for frame_idx in key_frame_indices:
        if frame_idx >= len(all_frames):
            print(f"警告: 帧索引 {frame_idx} 超出范围 (最大 {len(all_frames)-1})")
            continue
            
        try:
            # 提取图数据
            frame_data = extract_graph_data(all_frames[frame_idx], close_residues)
            
            # 计算边特征
            edges = calculate_edge_features(frame_data)
            
            # 输出边数信息
            res_res_edges = sum(1 for e in edges if e['type'] == 'res-res')
            res_sub_edges = sum(1 for e in edges if e['type'] == 'res-sub')
            print(f"帧 {frame_idx}: 残基-残基边数 = {res_res_edges}, 残基-底物边数 = {res_sub_edges}, 总边数 = {len(edges)}")
            
            # 获取节点特征
            # 残基节点特征
            residue_features_frame = residue_features_all[frame_idx]
            
            # 底物节点特征（全零）
            num_substrate = frame_data['num_substrate']
            substrate_features = np.zeros((num_substrate, 8))
            
            # 拼接节点特征
            node_features = np.vstack([residue_features_frame, substrate_features])
            print(f"节点特征形状: {node_features.shape}")
            
            # 保存帧结果
            key_frame_results.append({
                'frame_idx': frame_idx,
                'edges': edges,  # 边特征
                'node_features': node_features,  # 节点特征
                'num_residues': frame_data['num_residues'],  # 残基节点数
                'num_substrate': num_substrate  # 底物节点数
            })
        except Exception as e:
            print(f"处理帧 {frame_idx} 时出错: {e}")
    
    return key_frame_results

def process_all_pdbs(pdb_dir, key_frames_file, close_residues_file, output_dir, residue_features_dir):
    """
    处理目录中的所有PDB文件的关键帧
    """
    # 加载关键帧索引
    if not os.path.exists(key_frames_file):
        print(f"错误: 未找到关键帧文件 {key_frames_file}")
        return
    
    key_frames_array = np.load(key_frames_file)
    print(f"加载关键帧文件: {key_frames_file}, 形状: {key_frames_array.shape}")
    
    # 加载close_residues
    if not os.path.exists(close_residues_file):
        print(f"错误: 未找到close_residues文件 {close_residues_file}")
        return
    
    close_residues = np.load(close_residues_file)
    print(f"加载close_residues文件: {close_residues_file}, 包含 {len(close_residues)} 个残基")
    
    # 获取所有PDB文件
    pdb_files = []
    for f in os.listdir(pdb_dir):
        if f.endswith('.pdb'):
            # 尝试匹配数字文件名 (如 "1.pdb", "2.pdb")
            match = re.match(r'^(\d+)\.pdb$', f)
            if match:
                file_num = int(match.group(1))
                pdb_files.append((file_num, f))
    
    # 按数字排序
    pdb_files.sort(key=lambda x: x[0])
    pdb_files = [f[1] for f in pdb_files]
    
    print(f"找到 {len(pdb_files)} 个PDB文件")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个PDB文件
    for idx, pdb_file in enumerate(pdb_files):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        print(f"处理文件 {idx+1}/{len(pdb_files)}: {pdb_file}")
        
        # 获取该样本的关键帧索引
        if idx < len(key_frames_array):
            key_frame_indices = key_frames_array[idx]
        else:
            print(f"警告: 没有为 {pdb_file} 找到关键帧索引, 使用默认帧")
            key_frame_indices = list(range(20))
        
        # 处理关键帧
        key_frame_results = process_key_frames_for_pdb(
            pdb_path, key_frame_indices, close_residues, residue_features_dir
        )
        
        # 保存结果
        output_file = os.path.join(output_dir, f"{idx+1}_graph.npy")
        np.save(output_file, key_frame_results)
        print(f"已保存关键帧图数据: {output_file}, 包含 {len(key_frame_results)} 个关键帧")
    
    print("所有PDB文件处理完成!")

# 使用示例
if __name__ == "__main__":
    # 配置参数 - 使用相对路径
    PDB_DIR = "data/pdb"  # PDB文件目录
    KEY_FRAMES_FILE = "key_frames.npy"  # 关键帧索引文件
    CLOSE_RESIDUES_FILE = "data/close_residues.npy"  # 关注的残基文件
    OUTPUT_DIR = "data/graph"  # 输出目录
    RESIDUE_FEATURES_DIR = "data"  # 残基特征文件目录
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 处理所有PDB文件
    process_all_pdbs(PDB_DIR, KEY_FRAMES_FILE, CLOSE_RESIDUES_FILE, OUTPUT_DIR, RESIDUE_FEATURES_DIR)