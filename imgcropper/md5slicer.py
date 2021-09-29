import hashlib
import shutil
from pathlib import Path
from tqdm import tqdm


def get_img_hash_dict(img_list):
    """
    将文件路径通过md5加密
    return [md5 encode],{md5 encode:'img_path'}
    """
    hash_list = []
    result_neg = {}
    md5 = hashlib.md5()
    for img_path in img_list:
        md5.update(img_path.encode('utf-8'))
        hash_list.append(md5.hexdigest())
        result_neg[md5.hexdigest()]=img_path
    return hash_list,result_neg


def split_data(img_root,split_out_root):
    train_img_root = r'D:\data\mycrop\WhuAfter\train'
    split_out_root = r'D:\data\Whu\after'
    (Path(split_out_root) / "train" / "img").mkdir(exist_ok=True, parents=True)
    (Path(split_out_root) / "train" / "label").mkdir(exist_ok=True, parents=True)
    (Path(split_out_root) / "test" / "img").mkdir(exist_ok=True, parents=True)
    (Path(split_out_root) / "test" / "label").mkdir(exist_ok=True, parents=True)
    img_names = [str(f).replace('\\', '/').split('/')[-1] for f in (Path(train_img_root)/"img").iterdir()]
    img_hash_list, img_hash_dict= get_img_hash_dict(img_names)
    data_size = len(img_names)
    test_ratio = 0.1
    split_index = int(data_size*(1-test_ratio))
    sorted_img_dict = sorted(img_hash_list) # 通过img_path的md5值对img_path进行排序
    train_img_names = [img_hash_dict.get(hash_val) for hash_val in sorted_img_dict[:split_index]]
    test_imgs_names = [img_hash_dict.get(hash_val) for hash_val in sorted_img_dict[split_index:]]
    for train_name in tqdm(train_img_names):
        shutil.copy(str(Path(train_img_root)/"img"/train_name), str(Path(split_out_root) / "train" / "img"))
        shutil.copy(str(Path(train_img_root)/"label"/train_name), str(Path(split_out_root) / "train" / "label"))
    for test_name in tqdm(test_imgs_names):
        shutil.copy(str(Path(train_img_root)/"img"/test_name), str(Path(split_out_root) / "test" / "img"))
        shutil.copy(str(Path(train_img_root)/"label"/test_name), str(Path(split_out_root) / "test" / "label"))


if __name__ == '__main__':
    after_params = {
        "img_root": r'D:\data\mycrop\WhuAfter\train',
        "split_out_root": r'D:\data\Whu\after'
    }
    before_params = {
        "img_root": r'D:\data\mycrop\WhuBefore\train',
        "split_out_root": r'D:\data\Whu\before'
    }
    split_data(**after_params)
    split_data(**before_params)
