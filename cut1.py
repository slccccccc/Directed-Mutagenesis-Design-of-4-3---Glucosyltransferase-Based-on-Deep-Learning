import os
import numpy as np
import glob

# 特殊残基列表（催化三联体）
SPECIAL_RESIDUES = {372, 410, 482}

def parse_pdb_file(file_path):
    """解析PDB文件，返回所有帧的原子信息"""
    frames = []
    current_frame = []
    in_model = False
    
    with open(file_path, 'r') as f:
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
                    # 如果没有MODEL/ENDMDL，整个文件作为一个帧
                    if not frames:
                        frames.append([])
                    frames[0].append(line)
                else:
                    current_frame.append(line)
    
    # 如果没有找到任何帧，但文件中有ATOM行，则添加一个帧
    if not frames and current_frame:
        frames.append(current_frame)
    
    return frames

def get_atom_info(line):
    """从PDB行中提取原子信息"""
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    residue_number = int(line[22:26].strip())
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    
    return {
        'atom_name': atom_name,
        'residue_name': residue_name,
        'residue_number': residue_number,
        'coords': np.array([x, y, z])
    }

def find_close_residues(frames, existing_residues):
    """找出所有与底物或特殊残基指定原子在9埃范围内的残基"""
    close_residues = set()
    
    for frame in frames:
        # 收集底物H原子和特殊残基指定原子的坐标
        substrate_coords = []
        special_coords = []
        
        # 先收集所有相关坐标
        for line in frame:
            atom_info = get_atom_info(line)
            residue_number = atom_info['residue_number']
            atom_name = atom_info['atom_name']
            
            # 底物原子（残基979）
            if residue_number == 979:
                substrate_coords.append(atom_info['coords'])
            
            # 特殊残基的指定原子
            elif residue_number in SPECIAL_RESIDUES:
                special_coords.append(atom_info['coords'])
        
        # 合并所有关注点的坐标
        focus_coords = special_coords + substrate_coords
        
        for line in frame:
            atom_info = get_atom_info(line)
            residue_number = atom_info['residue_number']
            coords = atom_info['coords']
            
            # 跳过底物和特殊残基本身
            if residue_number == 979 or residue_number in existing_residues:
                continue
            
            # 检查距离
            for focus_coord in focus_coords:
                distance = np.linalg.norm(coords - focus_coord)
                if distance < 9:  # 9埃
                    # 记录残基号
                    close_residues.add(residue_number)
                    break  # 找到一个就跳出内层循环
    
    return close_residues

def main():
    # 设置目标目录
    target_dir = "D:/python/classification/data/pdb"
    output_dir = "D:/python/classification/data"  # 输出目录
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有处理后的PDB文件
    pdb_files = glob.glob(os.path.join(target_dir, "*.pdb"))
    
    # 全局残基集合
    global_close_residues = set()

    global_close_residues.update(SPECIAL_RESIDUES)

    print(f"开始处理 {len(pdb_files)} 个PDB文件...")
    
    for i, file_path in enumerate(pdb_files):
        filename = os.path.basename(file_path)
        print(f"处理文件 {i+1}/{len(pdb_files)}: {filename}")
        
        # 解析PDB文件
        frames = parse_pdb_file(file_path)
        
        # 找出接近的残基（传入当前全局集合以避免重复计算）
        close_residues = find_close_residues(frames, global_close_residues)
        
        # 添加到全局集合
        global_close_residues.update(close_residues)
        
        # 显示当前进度
        print(f"  已找到 {len(global_close_residues)} 个残基")
    
    # 转换为排序后的列表
    sorted_residues = sorted(global_close_residues)
    
    # 转换为numpy数组
    residues_array = np.array(sorted_residues)
    
    # 保存为npy文件
    output_path = os.path.join(output_dir, "close_residues.npy")
    np.save(output_path, residues_array)
    
    print("\n找到的接近残基列表:")
    for residue in sorted_residues:
        print(residue)
    
    print(f"\n总共找到 {len(sorted_residues)} 个接近残基")
    print(f"其中包括催化三联体: {SPECIAL_RESIDUES}")
    print(f"残基编号已保存到: {output_path}")

if __name__ == "__main__":
    main()