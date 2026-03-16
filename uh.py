import os
import numpy as np
import re
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)  # 抑制FutureWarning


def process_npy_files(data_dir, output_file, n_keyframes=20):
    """
    处理data目录下的数字编号npy文件，为每个样本找到指定数量的关键帧

    参数:
        data_dir: 包含npy文件的目录路径
        output_file: 输出结果保存的文件路径
        n_keyframes: 每个样本需要提取的关键帧数量，默认为20
    """
    # 获取所有数字编号的npy文件并按数字排序
    npy_files = []
    for f in os.listdir(data_dir):
        if f.endswith('.npy'):
            # 尝试匹配数字文件名 (如 "1.npy", "2.npy")
            match = re.match(r'^(\d+)\.npy$', f)
            if match:
                file_num = int(match.group(1))
                npy_files.append((file_num, f))

    # 按数字排序
    npy_files.sort(key=lambda x: x[0])
    npy_files = [f[1] for f in npy_files]  # 只保留文件名

    # 存储所有样本的关键帧索引
    all_key_frames = []

    print(f"找到 {len(npy_files)} 个数字编号的npy文件，开始处理...")
    print(f"目标关键帧数量: {n_keyframes}")

    for file in tqdm(npy_files, desc="处理样本"):
        file_path = os.path.join(data_dir, file)

        try:
            # 加载数据 [帧数, 残基数, 点特征维度]
            data = np.load(file_path)
            n_frames, n_residues, n_features = data.shape

            print(f"处理文件 {file}: 帧数={n_frames}, 残基数={n_residues}, 特征维度={n_features}")

            # 如果帧数不足n_keyframes，使用所有帧作为关键帧
            if n_frames <= n_keyframes:
                key_frames = list(range(n_frames))
                # 如果帧数不足n_keyframes，重复使用现有帧补足
                while len(key_frames) < n_keyframes:
                    key_frames.append(key_frames[-1])  # 重复最后一帧
                all_key_frames.append(key_frames[:n_keyframes])  # 确保不超过n_keyframes帧
                continue

            # 1. 重塑数据为 [帧数, 残基数*点特征维度]
            reshaped_data = data.reshape(n_frames, n_residues * n_features)

            # 2. 改进的标准化 - 使用RobustScaler处理异常值
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(reshaped_data)

            # 3. 调整UMAP降维参数 - 增加保留的结构信息
            reducer = umap.UMAP(
                n_components=min(10, n_frames // n_keyframes),  # 增加维度以保留更多结构
                n_neighbors=min(15, max(5, n_frames // 30)),  # 减少邻居数以捕获局部结构
                min_dist=0.1,  # 减少最小距离以形成更密集的嵌入
                random_state=42,
                metric='cosine'  # 使用余弦距离
            )
            embedding = reducer.fit_transform(scaled_data)

            # 4. 关键修改: 调整HDBSCAN参数以生成更多簇
            min_cluster_size = max(5, n_frames // 100)  # 显著减少最小簇大小
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=2,  # 减少最小样本数
                cluster_selection_method='eom',  # 使用eom方法而不是leaf，以生成更多簇
                cluster_selection_epsilon=0.3,  # 调整选择阈值
                gen_min_span_tree=True  # 生成最小生成树以提高稳定性
            )
            cluster_labels = clusterer.fit_predict(embedding)

            # 5. 为每个簇找到中心帧
            unique_labels = np.unique(cluster_labels)
            key_frames = []

            # 排除噪声点簇 (标签为-1)
            if -1 in unique_labels:
                unique_labels = unique_labels[unique_labels != -1]

            print(f"  生成簇数量: {len(unique_labels)}")

            # 为每个簇找到中心帧
            for label in unique_labels:
                # 获取当前簇的所有点
                cluster_mask = (cluster_labels == label)
                cluster_points = embedding[cluster_mask]

                # 计算簇中心
                cluster_center = np.mean(cluster_points, axis=0)

                # 找到距离中心最近的帧
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                closest_idx_in_cluster = np.argmin(distances)

                # 转换为原始帧索引
                original_indices = np.where(cluster_mask)[0]
                key_frame_idx = original_indices[closest_idx_in_cluster]
                key_frames.append(key_frame_idx)

            # 6. 确保每个样本有n_keyframes个关键帧
            if len(key_frames) > n_keyframes:
                # 如果簇太多，取前n_keyframes个最大的簇
                cluster_sizes = []
                for label in unique_labels:
                    cluster_sizes.append(np.sum(cluster_labels == label))

                sorted_indices = np.argsort(cluster_sizes)[::-1]  # 从大到小排序
                key_frames = [key_frames[i] for i in sorted_indices[:n_keyframes]]
            elif len(key_frames) < n_keyframes:
                # 如果簇太少，添加一些补充帧
                n_missing = n_keyframes - len(key_frames)
                print(f"  簇数量不足: {len(key_frames)}，需要补充 {n_missing} 帧")

                # 添加整个轨迹中变化最大的帧
                frame_variation = np.std(embedding, axis=1)

                # 确保选择变化大且时间分散的帧
                selected_frames = []
                step = max(1, n_frames // n_missing)  # 计算步长

                for i in range(n_missing):
                    # 在每个时间区间内选择变化最大的帧
                    start = i * step
                    end = min((i + 1) * step, n_frames)

                    if end > start:  # 确保区间有效
                        segment_variation = frame_variation[start:end]

                        if len(segment_variation) > 0:
                            max_var_idx = np.argmax(segment_variation)
                            frame_idx = start + max_var_idx
                            if frame_idx not in key_frames and frame_idx not in selected_frames:
                                selected_frames.append(frame_idx)

                key_frames.extend(selected_frames)

                # 如果还不够，再添加一些随机帧
                if len(key_frames) < n_keyframes:
                    n_needed = n_keyframes - len(key_frames)
                    # 从非关键帧中随机选择
                    non_key_frames = list(set(range(n_frames)) - set(key_frames))
                    if len(non_key_frames) > 0:
                        random_frames = np.random.choice(
                            non_key_frames,
                            min(n_needed, len(non_key_frames)),
                            replace=False
                        )
                        key_frames.extend(random_frames.tolist())

                # 如果仍然不足，重复使用现有帧
                if len(key_frames) < n_keyframes:
                    n_needed = n_keyframes - len(key_frames)
                    for i in range(n_needed):
                        key_frames.append(key_frames[i % len(key_frames)])

            # 对关键帧排序并确保唯一性
            key_frames = sorted(set(key_frames))
            if len(key_frames) > n_keyframes:
                # 如果超过n_keyframes帧，选择时间上分布均匀的n_keyframes帧
                key_frames = [key_frames[i] for i in np.linspace(0, len(key_frames) - 1, n_keyframes, dtype=int)]

            all_key_frames.append(key_frames)

        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            # 如果出错，使用等间隔帧作为关键帧
            try:
                total_frames = data.shape[0] if 'data' in locals() else 100
                key_frames = []
                for i in range(n_keyframes):
                    frame_index = int(total_frames * i / n_keyframes)  # 等间隔选择
                    frame_index = max(0, min(frame_index, total_frames - 1))
                    key_frames.append(frame_index)
                all_key_frames.append(key_frames)
                print(f"  使用等间隔帧作为关键帧: {key_frames}")
            except:
                # 如果连等间隔选择也失败，使用默认值
                all_key_frames.append(list(range(n_keyframes)))
                print(f"  使用默认关键帧: {list(range(n_keyframes))}")

    # 转换为二维数组 [样本数, n_keyframes]
    key_frames_array = np.array(all_key_frames)

    # 验证结果
    print(f"处理完成! 结果数组形状: {key_frames_array.shape}")
    print(f"每个样本的关键帧数量统计:")
    frame_counts = [len(set(frames)) for frames in key_frames_array]
    print(f"  平均关键帧数: {np.mean(frame_counts):.2f}")
    print(f"  最小关键帧数: {np.min(frame_counts)}")
    print(f"  最大关键帧数: {np.max(frame_counts)}")
    print(f"  唯一关键帧比例: {np.mean([c / n_keyframes for c in frame_counts]):.2%}")

    # 保存结果
    np.save(output_file, key_frames_array)
    print(f"结果已保存至 {output_file}")

    return key_frames_array


def validate_keyframes(key_frames_array, data_dir, n_keyframes=20):
    """
    验证关键帧的质量

    参数:
        key_frames_array: 关键帧数组
        data_dir: 数据目录路径
        n_keyframes: 每个样本的关键帧数量
    """
    print(f"\n关键帧质量验证 (目标数量: {n_keyframes}):")

    # 1. 检查关键帧是否在有效范围内
    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    npy_files.sort()

    valid_count = 0
    for i, file in enumerate(npy_files[:min(5, len(npy_files))]):  # 检查前5个文件
        file_path = os.path.join(data_dir, file)
        try:
            data = np.load(file_path)
            n_frames = data.shape[0]
            key_frames = key_frames_array[i]

            # 检查关键帧是否在有效范围内
            valid_frames = [f for f in key_frames if 0 <= f < n_frames]
            validity_ratio = len(valid_frames) / len(key_frames)

            if validity_ratio == 1.0:
                valid_count += 1
                status = "✓ 有效"
            else:
                status = f"✗ 无效 ({validity_ratio:.1%})"

            print(f"  文件 {file}: {status} ({len(valid_frames)}/{len(key_frames)} 有效帧)")

        except Exception as e:
            print(f"  文件 {file}: 验证失败 - {e}")

    print(f"验证完成: {valid_count}/{min(5, len(npy_files))} 个文件关键帧完全有效")


if __name__ == "__main__":
    # 配置参数
    DATA_DIR = "data"  # 包含npy文件的目录
    OUTPUT_FILE = "key_frames.npy"  # 输出文件路径
    N_KEYFRAMES = 20  # 每个样本需要提取的关键帧数量，可按需修改

    # 执行处理
    key_frames_result = process_npy_files(DATA_DIR, OUTPUT_FILE, n_keyframes=N_KEYFRAMES)

    # 验证关键帧质量
    validate_keyframes(key_frames_result, DATA_DIR, n_keyframes=N_KEYFRAMES)

    print(f"\n关键帧提取流程完成! 每个样本提取了 {N_KEYFRAMES} 个关键帧。")
