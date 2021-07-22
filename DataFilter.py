import json

# with open("./Data/NYBI_TRAIN.json", 'r', encoding='utf-8') as ft, open("./Data/NYBI_PRED.json", 'r',
#                                                                        encoding='utf-8') as fp:
    # data = ft.readlines()
    # count = 0
    # summary_freq = dict()
    # for i, line in enumerate(data):
    #     dec_json = json.loads(line)
    #     english_summary = dec_json['chinese_summary']
    #     if english_summary not in summary_freq:
    #         summary_freq[english_summary] = 1
    # print(f"TRAIN 文件里面共有不同的数据{len(summary_freq)}条")
    # pred_data = fp.readlines()
    # for p_line in pred_data:
    #     dec_json = json.loads(p_line)
    #     english_summary = dec_json['chinese_summary']
    #     if english_summary in summary_freq:
    #         count += 1
    #     else:
    #         print(english_summary)
    # print(f"pred文件里面共有{count}条与train文件中重复的数据")

with open("./Data/NYTimes_Bi_Raw_transed.json",'r',encoding='utf-8') as f, open("./Data/NYTimes_Bi_Trans_Clean.json",'w',encoding='utf-8') as fp:
    data = f.readlines()
    count = 0
    summary_freq = dict()
    for i, line in enumerate(data):
        flag = False
        dec_json = json.loads(line)
        summary_dict = dec_json['Summary']
        content_list = dec_json['Content']
        en_sum = summary_dict['english']
        if en_sum == "Here’s what you need to know to start your day.":
            count += 1
            continue
        for d in content_list:
            chinese_sent = d['chinese']
            english_sent = d['english']
            if (chinese_sent and not english_sent) or (not chinese_sent and english_sent):
                flag = True
                break
        if flag:
            count += 1
            continue
        if en_sum not in summary_freq:
            summary_freq[en_sum] = 1
            print(json.dumps(dec_json, ensure_ascii=False), file=fp)
        else:
            count += 1
            continue
    print("清洗完毕，一共清洗掉{}条数据".format(count))
