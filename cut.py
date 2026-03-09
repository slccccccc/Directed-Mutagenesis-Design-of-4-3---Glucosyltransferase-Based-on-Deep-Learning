import os
import argparse
from multiprocessing import Pool
import glob

# 氨基酸与保留原子的映射
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

def should_keep_atom(residue_name, atom_name):
    """判断是否应该保留此原子"""
    # 检查残基名是否在映射表中
    if residue_name in RESIDUE_ATOM_MAP:
        # 检查原子名是否匹配
        return RESIDUE_ATOM_MAP[residue_name] == atom_name.strip()
    return False

def process_pdb_file(file_pair):
    """处理单个PDB文件"""
    source_path, target_path = file_pair
    
    preserved_lines = []
    current_residue_number = None
    current_residue_name = None
    
    with open(source_path, 'r') as f:
        for line in f:
            # 保留所有非ATOM行（MODEL, ENDMDL等）
            if not line.startswith("ATOM"):
                preserved_lines.append(line)
                continue
            
            # 提取残基信息
            residue_number = line[22:26].strip()
            residue_name = line[17:20].strip()
            atom_name = line[12:16].strip()
            
            # 处理底物分子（残基979） - 只保留碳(C)和氧(O)原子
            if residue_number == "979":
                # 检查原子名称的第一个字符
                if atom_name[0] in ['C', 'O']:
                    preserved_lines.append(line)
                continue
            
            # 处理酶分子
            # 检查是否应该保留此原子
            if should_keep_atom(residue_name, atom_name):
                preserved_lines.append(line)
    
    # 写入目标文件
    with open(target_path, 'w') as f:
        f.writelines(preserved_lines)
    
    return 1

def main():
    parser = argparse.ArgumentParser('PDB裁剪工具')
    parser.add_argument('--source-dir', type=str, required=True, help='源PDB文件夹路径')
    parser.add_argument('--target-dir', type=str, required=True, help='目标PDB文件夹路径')
    parser.add_argument('--processes', type=int, default=8, help='并行进程数')
    args = parser.parse_args()

    # 确保目标目录存在
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir, exist_ok=True)
    
    # 获取所有PDB文件路径
    source_files = glob.glob(os.path.join(args.source_dir, '*.pdb'))
    file_pairs = []
    for source_file in source_files:
        filename = os.path.basename(source_file)
        target_path = os.path.join(args.target_dir, filename)
        file_pairs.append((source_file, target_path))

    # 并行处理
    with Pool(args.processes) as pool:
        results = pool.map(process_pdb_file, file_pairs)
        processed_files = sum(results)

    print(f"成功裁剪 {processed_files} 个PDB文件！")
    print(f"源目录: {args.source_dir}")
    print(f"目标目录: {args.target_dir}")
    print("裁剪规则：")
    print("1. 底物分子（979号残基）只保留碳(C)和氧(O)原子")
    print("2. 酶分子根据以下规则保留特定原子：")
    for res, atom in RESIDUE_ATOM_MAP.items():
        print(f"   {res}: {atom}")

if __name__ == "__main__":
    main()
#python cut.py --source-dir "H:\2970 mutant——rj\all" --target-dir "D:\python\classification\data\pdb" --processes 8
#python cut.py --source-dir "G:\all" --target-dir "D:\python\classification\data\pdb" --processes 8