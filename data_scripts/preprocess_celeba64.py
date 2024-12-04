import os  # 导入操作系统模块，用于文件和目录操作
import pandas as pd  # 导入 pandas 库，用于数据处理
from PIL import Image  # 导入 PIL 库中的 Image 模块，用于图像处理
from functools import partial  # 导入 partial，用于创建部分函数
from concurrent.futures import ProcessPoolExecutor  # 导入进程池执行器，用于并行处理
from tqdm import tqdm  # 导入 tqdm，用于显示进度条

# 定义数据集存放的根目录
dump_folder = '../dataset_dumps/'

def read_and_crop(image_filename):
    """
    读取并裁剪图像
    :param image_filename: 图像文件名
    :return: 裁剪后的图像
    """
    # 打开图像并转换为 RGB 格式，然后裁剪指定区域
    return Image.open(os.path.join(dump_folder, 'img_align_celeba', image_filename)).convert("RGB").crop((15, 40, 178 - 15, 218 - 30))

def save(path, image_filename, image):
    """
    保存图像到指定路径
    :param path: 保存路径
    :param image_filename: 图像文件名
    :param image: 要保存的图像
    """
    # 将图像保存到指定路径
    image.save(os.path.join(path, image_filename))

def process_one_file(image_filename, split):
    """
    处理单个图像文件，包括读取、裁剪和保存
    :param image_filename: 图像文件名
    :param split: 数据集划分（训练、验证或测试）
    """
    # 创建输出文件夹，如果不存在则创建
    output_folder = os.path.join('../datasets/celebA', split)
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取并裁剪图像
    image = read_and_crop(image_filename)
    
    # 保存裁剪后的图像
    save(output_folder, image_filename, image)

def get_partitioned_filenames():
    """
    获取分区后的文件名列表（训练、验证和测试）
    :return: 训练、验证和测试文件名的元组
    """
    # 读取分区文件，使用空格作为分隔符，并指定列名
    partition_list = pd.read_csv(os.path.join(dump_folder, 'list_eval_partition.txt'), sep=' ', names=['filenames', 'partition'])
    
    # 根据分区信息获取不同数据集的文件名
    train_filenames = partition_list[partition_list.partition == 0]
    val_filenames = partition_list[partition_list.partition == 1]
    test_filenames = partition_list[partition_list.partition == 2]
    
    # 返回训练、验证和测试文件名
    return train_filenames.filenames, val_filenames.filenames, test_filenames.filenames

def main():
    """
    主函数，负责处理所有图像文件
    """
    # 定义数据集划分
    splits = ['train_data', 'val_data', 'synthesis_data']
    
    # 获取分区后的文件名
    all_filenames = get_partitioned_filenames()
    
    # 遍历每个划分及其对应的文件名
    for split, split_filenames in zip(splits, all_filenames):
        print(f'Processing {split}..')  # 打印当前处理的划分
        
        # 创建进程池执行器，最大工作进程数为 256
        executor = ProcessPoolExecutor(max_workers=256)
        
        # 提交任务到进程池，使用 partial 创建部分函数
        futures = [executor.submit(partial(process_one_file, filename, split)) for filename in split_filenames]
        
        # 等待所有任务完成，并显示进度条
        _ = [future.result() for future in tqdm(futures)]

if __name__ == '__main__':
    main()  # 运行主函数
