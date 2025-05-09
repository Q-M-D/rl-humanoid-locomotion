import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_data(data, save_path=None):
    # 设置matplotlib后端为Agg，这样不会创建GUI窗口
    plt.switch_backend('Agg')
    
    # 检查数据结构
    if not isinstance(data, dict):
        print("Error: Data is not in dictionary format")
        return
    
    # 为每个主键创建单独的图形
    for key in data.keys():
        try:
            # 获取当前键的数据
            current_data = data[key]
            
            # 检查数据类型
            if isinstance(current_data, dict):
                # 如果是字典，获取子键
                sub_keys = list(current_data.keys())
                num_subplots = len(sub_keys)
                
                # 创建图形
                fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 3*num_subplots))
                if num_subplots == 1:
                    axs = [axs]  # 确保axs是列表
                
                # 绘制每个子图
                for j, sub_key in enumerate(sub_keys):
                    values = current_data[sub_key]
                    if isinstance(values, list):
                        # 如果数据太长，进行下采样
                        if len(values) > 10000:
                            step = len(values) // 10000
                            values = values[::step]
                        
                        # 绘制数据
                        axs[j].plot(values)
                        axs[j].set_title(f"{key} - {sub_key}")
                        axs[j].grid(True)
                        
                        # 如果数据量很大，只显示部分x轴标签
                        if len(values) > 100:
                            axs[j].set_xticks(axs[j].get_xticks()[::2])
                    else:
                        print(f"Warning: Data for {key} - {sub_key} is not a list")
            
            elif isinstance(current_data, list):
                # 如果是列表，直接绘制
                fig, ax = plt.subplots(figsize=(10, 6))
                values = current_data
                
                # 如果数据太长，进行下采样
                if len(values) > 10000:
                    step = len(values) // 10000
                    values = values[::step]
                
                ax.plot(values)
                ax.set_title(key)
                ax.grid(True)
                
                if len(values) > 100:
                    ax.set_xticks(ax.get_xticks()[::2])
            
            # 调整布局
            plt.tight_layout()
            
            # 保存或显示图形
            if save_path:
                plt.savefig(os.path.join(save_path, f'{key}.png'), dpi=300, bbox_inches='tight')
            else:
                plt.show()
            
            # 关闭图形以释放内存
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
            continue

if __name__ == "__main__":
    # 设置数据路径
    data_path = "/home/mmlab-rl/codes/sensorimotor-rl/sensorimotor/data/forward_0.313.json"
    save_path = os.path.join(os.path.dirname(data_path), "plots")
    
    # 创建保存目录
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    try:
        # 加载数据
        data = load_data(data_path)
        
        # 打印数据结构信息
        print("Data structure:")
        for key in data.keys():
            print(f"{key}: {type(data[key])}")
            if isinstance(data[key], dict):
                print(f"  Subkeys: {list(data[key].keys())}")
        
        # 绘制数据
        plot_data(data, save_path)
    except Exception as e:
        print(f"Error: {str(e)}")