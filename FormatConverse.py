import json
from tqdm import tqdm

with open("./Data/NYBI_TRAIN_TRANS.json", 'w', encoding='utf-8') as ft, \
        open("./Data/NYBI_EVAL_TRANS.json", 'w', encoding='utf-8') as fe, \
        open("./Data/NYBI_PRED_TRANS.json", 'w', encoding='utf-8') as fp, \
        open("./Data/NYTimes_Bi_TRANS_Clean.json", 'r', encoding='utf-8') as fs:
    data = fs.readlines()
    for i, line in tqdm(enumerate(data)):
        f = fe if i < 500 else (fp if i < 1000 else ft)
        try:
            dec_json = json.loads(line)
        except json.decoder.JSONDecodeError as e:
            print(f"行{i}进行解析时出错，错误信息{e}")
            continue
        content_list = dec_json['Content']
        summary_dict = dec_json['Summary']
        enc_dict = dict()
        enc_dict["english_summary"] = summary_dict['english']
        enc_dict["chinese_summary"] = summary_dict['chinese']
        english_content = ""
        chinese_content = ""

        for d in content_list:
            english_content += d['mt_english']
            chinese_content += d['mt_chinese']
        enc_dict['english_content'] = english_content
        enc_dict['chinese_content'] = chinese_content
        enc_json = json.dumps(enc_dict, ensure_ascii=False)
        print(enc_json, file=f)
