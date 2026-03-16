import os
import numpy as np
import glob
import argparse
import time
import re

# 残基到关键原子的映射
RESIDUE_ATOM_MAP = {
    'ASP': 'OD2',
    'GLU': 'OE2',
    'LYS': 'NZ',
    'ARG': 'CZ',
    'ASN': 'ND2',
    'GLN': 'NE2',
    'SER': 'OG',
    'THR': 'OG1',
    'TYR': 'OH',
    'HIE': 'ND1',
    'HIS': 'ND1',
    'CYS': 'SG',
    'PHE': 'CZ',
    'TRP': 'NE1',
    'ALA': 'CB',
    'VAL': 'CB',
    'LEU': 'CG',
    'ILE': 'CB',
    'MET': 'SD',
    'PRO': 'CG',
    'GLY': 'CA'
}

def natural_sort_key(s):
    """自然排序键函数，用于按数字顺序排序文件名"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

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

def calc_dihedral(p1, p2, p3, p4):
    """计算四个点形成的二面角（单位：弧度）"""
    try:
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        p4 = np.array(p4)
        
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        b2 /= np.linalg.norm(b2)
        
        angle = np.arctan2(np.dot(np.cross(n1, n2), b2), np.dot(n1, n2))
        return angle
    except:
        return 0.0

def calculate_features_for_frame(frame_lines, target_residues):
    """计算单个帧的phi/psi特征"""
    # 提取所有原子的坐标并分组
    atom_data = {}
    residue_keys = set()
    extended_residues = set()
    
    # 收集目标残基及其前后残基
    for res_id in target_residues:
        extended_residues.add(res_id - 1)
        extended_residues.add(res_id)
        extended_residues.add(res_id + 1)
    
    for line in frame_lines:
        atom_info = extract_atom_info(line)
        if atom_info is None:
            continue
            
        res_id = atom_info['residue_number']
        chain_id = atom_info['chain_id']
        
        # 只处理目标残基及其前后残基
        if res_id not in extended_residues:
            continue
            
        key = (chain_id, res_id)
        residue_keys.add(key)
        
        if key not in atom_data:
            atom_data[key] = {}
        
        atom_name = atom_info['atom_name']
        atom_data[key][atom_name] = atom_info
    
    # 初始化结果字典
    phi_psi_map = {}
    
    # 计算phi/psi角
    for key in residue_keys:
        chain_id, res_id = key
        curr_atoms = atom_data.get(key, {})
        prev_atoms = atom_data.get((chain_id, res_id - 1), {})
        next_atoms = atom_data.get((chain_id, res_id + 1), {})
        
        # 计算phi角
        if 'C' in prev_atoms and 'N' in curr_atoms and 'CA' in curr_atoms and 'C' in curr_atoms:
            try:
                phi = calc_dihedral(
                    prev_atoms['C']['coords'],
                    curr_atoms['N']['coords'],
                    curr_atoms['CA']['coords'],
                    curr_atoms['C']['coords']
                )
                phi_sin = np.sin(phi)
                phi_cos = np.cos(phi)
            except:
                phi_sin, phi_cos = 0.0, 1.0
        else:
            phi_sin, phi_cos = 0.0, 1.0
        
        # 计算psi角
        if 'N' in curr_atoms and 'CA' in curr_atoms and 'C' in curr_atoms and 'N' in next_atoms:
            try:
                psi = calc_dihedral(
                    curr_atoms['N']['coords'],
                    curr_atoms['CA']['coords'],
                    curr_atoms['C']['coords'],
                    next_atoms['N']['coords']
                )
                psi_sin = np.sin(psi)
                psi_cos = np.cos(psi)
            except:
                psi_sin, psi_cos = 0.0, 1.0
        else:
            psi_sin, psi_cos = 0.0, 1.0
        
        phi_psi_map[key] = [phi_sin, phi_cos, psi_sin, psi_cos]
    
    return phi_psi_map

def process_pdb_file(pdb_path, output_dir, target_residues):
    """处理单个PDB文件，计算所有特征"""
    try:
        # 获取文件名（不含扩展名）
        filename = os.path.basename(pdb_path).replace('.pdb', '')
        start_time = time.time()
        
        # 解析PDB文件（文本方式）
        frames = parse_pdb_file(pdb_path)
        if not frames:
            print(f"  无法解析PDB文件: {filename}")
            return 0
            
        n_models = len(frames)
        print(f"  模型数: {n_models}")
        
        # 获取所有残基，但只保留在target_residues中的残基
        residue_keys = set()
        for line in frames[0]:
            atom_info = extract_atom_info(line)
            if atom_info is None:
                continue
            res_id = atom_info['residue_number']
            if res_id in target_residues:
                chain_id = atom_info['chain_id']
                residue_keys.add((chain_id, res_id))
        
        n_residues = len(residue_keys)
        
        # 初始化特征数组 (4维特征)
        features = np.zeros((n_models, n_residues, 4))
        
        # 为每个模型计算特征
        for model_id in range(n_models):
            # 计算phi/psi特征
            phi_psi_map = calculate_features_for_frame(frames[model_id], target_residues)
            
            # 将特征填充到数组中
            for i, key in enumerate(residue_keys):
                # Phi/Psi特征 (4维)
                phi_psi = phi_psi_map.get(key, [0.0, 1.0, 0.0, 1.0])
                features[model_id, i, 0:4] = phi_psi
        
        # 保存为npy文件
        output_path = os.path.join(output_dir, f"{filename}.npy")
        np.save(output_path, features)
        
        total_elapsed = time.time() - start_time
        print(f"处理完成: {filename}, 模型数: {n_models}, 残基数: {n_residues}, 总耗时: {total_elapsed:.2f}s")
        return 1
        
    except Exception as e:
        print(f"处理文件 {pdb_path} 时出错: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser('PDB特征提取工具')
    parser.add_argument('--pdb-dir', type=str, required=True, help='PDB文件夹路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出npy文件夹路径')
    parser.add_argument('--residues-file', type=str, required=True, help='残基编号列表文件路径（.npy文件）')
    parser.add_argument('--single-pdb', type=int, help='只处理指定的PDB文件编号（如：24）')
    args = parser.parse_args()

    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载残基编号列表
    target_residues = set(np.load(args.residues_file))
    print(f"加载了 {len(target_residues)} 个目标残基")
    
    # 获取所有PDB文件并按数字顺序排序
    pdb_files = glob.glob(os.path.join(args.pdb_dir, "*.pdb"))
    pdb_files.sort(key=natural_sort_key)
    
    # 如果指定了单个PDB文件，只处理该文件
    if args.single_pdb is not None:
        # 构造目标文件名
        target_filename = f"{args.single_pdb}.pdb"
        target_path = None
        
        # 在排序后的文件列表中查找目标文件
        for pdb_path in pdb_files:
            if os.path.basename(pdb_path) == target_filename:
                target_path = pdb_path
                break
        
        if target_path is not None:
            print(f"只处理PDB文件: {target_filename}")
            pdb_files = [target_path]
        else:
            print(f"未找到PDB文件: {target_filename}")
            return
    
    print(f"找到 {len(pdb_files)} 个PDB文件")
    
    total_start = time.time()
    processed_files = 0
    
    # 顺序处理每个PDB文件
    for pdb_path in pdb_files:
        result = process_pdb_file(pdb_path, args.output_dir, target_residues)
        processed_files += result
    
    total_elapsed = time.time() - total_start
    print(f"所有文件处理完成! 成功处理 {processed_files} 个PDB文件，总耗时: {total_elapsed:.2f}s")
    print(f"输出目录: {args.output_dir}")
    print(f"特征维度: [帧数, 残基数（过滤后）, 4]")
    print("特征说明:")
    print("  0: phi角的正弦值")
    print("  1: phi角的余弦值")
    print("  2: psi角的正弦值")
    print("  3: psi角的余弦值")

if __name__ == "__main__":
    main()
#python node2.py --pdb-dir "G:\all" --output-dir "D:\python\classification\data" --residues-file "D:\python\classification\data\close_residues.npy" --single-pdb 24

#python node2.py --pdb-dir "G:\all" --output-dir "D:\python\classification\data" --residues-file "D:\python\classification\data\close_residues.npy"
