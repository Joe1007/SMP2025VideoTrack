import json
import os
from datetime import datetime
import logging
import json
from typing import List, Dict, Set



# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIG = {
    'video_path_replace': '../raw_data/video_file/',  # 檔案路徑的前綴

    'test_post_path': '../raw_data/test/SMP-Video_anonymized_posts_test.jsonl',
    'test_users_path': '../raw_data/test/SMP-Video_anonymized_users_test.jsonl',
    'test_video_info_path': '../raw_data/test/SMP-Video_anonymized_videos_test.jsonl',

    'test_video_path': '../raw_data/video_file/test/',

    'output_dir': '.', 
    'train_users_path': '../raw_data/train/SMP-Video_anonymized_users_train.jsonl',
    
}



def clean(data):
    """
    清理資料，完成以下操作：
    1. 如果 'user_likes_count' 與 'user_heart_count' 的數值完全相同，保留 'user_likes_count'。
    2. 刪除 'user_friend_count' 是全0的資料。
    3. 修正 'user_likes_count' 中小於 0 的數值，將其加上 2^32。
    4. 將多種語言或拼寫的「原創音訊」統一為 'original sound'。
    """
    # 定義所有需要替換為 original sound 的標籤
    # 定義所有需要替換為 original sound 的標籤
    ORIGINAL_SOUND_LABELS = {
    "original sound",
    "sonido original",
    "som original",
    "son original",
    "suono originale",
    "sunet original",
    "sonus originalis",
    "صدای اصلی",
    "sonu originale",
    "Phokoso loyambirira",
    "origina sono",
    "оригінальний звук",
    "원본 소리",
    "originalus garsas",
    "Swara asli",
    "оригинальный звук",
    "sain wreiddiol",
    "son orixinal",
    "originalni zvuk",
    "orijinal ses",
    "orijinal səs",
    "oarspronklik lûd",
    "origineel geluid",
    "umsindo wokuqala",
    "օրիգինալ ձայն",
    "âm thanh gốc",
    "suara asli",
    "bunyi asal",
    "原聲",
    "orihinal na tunog",
    "原声",
    "dengê orîjînal",
    "izvorni zvuk",
    "الصوت الأصلي",
    "izvorni zvok",
    "eredeti hang",
    "เสียงต้นฉบับ",
    "zëri origjinal",
    "originalljud",
    "sora asli",
    "ruzha rwepakutanga"
    }



    cleaned_data = {}

    for record in data:
        # 0. normalize post_location: strip whitespace and uppercase
        if 'post_location' in record:
            record['post_location'] = record['post_location'].strip().upper()
        


        # 1. 處理 user_likes_count 與 user_heart_count 相同的情況
        if 'user_likes_count' in record and 'user_heart_count' in record:
            if record['user_likes_count'] == record['user_heart_count']:
                del record['user_heart_count']
            if record['user_likes_count'] < 0:
                record['user_likes_count'] += 2**32

        # 2. 刪除 user_friend_count 為全0的欄位
        if 'user_friend_count' in record and record['user_friend_count'] == 0:
            del record['user_friend_count']

        # 3. 統一 music_title 標籤
        mt = record.get('music_title', '').strip().lower()
        if mt in ORIGINAL_SOUND_LABELS:
            record['music_title'] = "original sound"

        cleaned_data[record.get('pid', str(record))] = record

    return cleaned_data


def load_and_combine(*paths):
    """
    接收多個檔案路徑 (jsonl格式)，將資料合併並處理。
    """
    combined_by_pid = {}    # key: pid, value: dict(內含該 pid 相關資訊)
    user_data_by_uid = {}   # key: uid, value: dict(內含該 uid 相關資訊)

    for path in paths:
        logger.info(f"正在處理檔案: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())

                # 處理 'pid' 的資料
                if 'pid' in record:
                    pid = record['pid']
                    if pid not in combined_by_pid:
                        combined_by_pid[pid] = {}
                    combined_by_pid[pid].update(record)

                # 處理 'uid' 的資料
                elif 'uid' in record:
                    uid = record['uid']
                    if uid not in user_data_by_uid:
                        user_data_by_uid[uid] = {}
                    user_data_by_uid[uid].update(record)

                else:
                    logger.warning(f"跳過無效資料行: {line}")

    logger.info(f"共讀取到 {len(combined_by_pid)} 筆包含 pid 的資料")
    logger.info(f"共讀取到 {len(user_data_by_uid)} 筆包含 uid 的資料")

    # 將 user-based 資料合併到 pid-based 資料
    for pid, pid_data in combined_by_pid.items():
        uid = pid_data.get('uid')
        if uid and uid in user_data_by_uid:
            combined_by_pid[pid].update(user_data_by_uid[uid])

    # 處理 'video_path' 變更為完整路徑
    logger.info("開始處理 'video_path' 為完整路徑")
    for pid, record in combined_by_pid.items():
        if 'video_path' in record:
            video_path = record['video_path']
            # 修改為完整路徑
            full_video_path = os.path.join(CONFIG['video_path_replace'], video_path)
            record['video_path'] = full_video_path

    return combined_by_pid #list(combined_by_pid.values())





def load_and_find_common_ids(train_path: str, test_path: str) -> Set:
    """
    讀取 train 與 test JSONL 檔案，並回傳同時出現在兩者的 uid 集合
    """
    train_ids: Set = set()
    test_ids: Set = set()
    for path, id_set in [(train_path, train_ids), (test_path, test_ids)]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if 'uid' in record:
                    id_set.add(record['uid'])
    print(f"共有{len(train_ids & test_ids)}個重複的uid")
    return train_ids & test_ids


def remark(data: Dict) -> Dict:
    """
    data: dict mapping pid -> record dict（需包含 'uid'）
    返回一個 dict，以 pid 為 key，Value 為 pioneer 標記 (0/1)
    """
    common_ids = load_and_find_common_ids(CONFIG['train_users_path'],CONFIG['test_users_path'])
    for pid, record in data.items():
        uid = record.get('uid')
        record['pioneer'] = 1 if uid in common_ids else 0
    return data



def main():
    merged_data = load_and_combine(
        CONFIG['test_post_path'],
        CONFIG['test_video_info_path'],
        CONFIG['test_users_path']
    )
    #print(merged_data[0])
    output_path = os.path.join(CONFIG['output_dir'], 'merged_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        for pid, record in merged_data.items():
            # 這裡寫入的是每一行一個完整的字典
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    merged_data=list(merged_data.values())
    # 清理資料
    
    #cleaned_data = clean(merged_data))
    cleaned_data = remark(clean(merged_data))


    # 確保輸出目錄存在
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print(f"清理後資料共 {len(cleaned_data)} 筆")

    # 儲存為 JSON 檔案
    output_path = os.path.join(CONFIG['output_dir'], 'test_cleaned_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        for pid, record in cleaned_data.items():
            # 這裡寫入的是每一行一個完整的字典
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"清理後資料已儲存至: {output_path}")



if __name__ == "__main__":
    main()
