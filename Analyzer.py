import json
import nltk
from transformers import BertTokenizer
from textrank4zh import util

def sent_segmentation(text):
    '''

    :param text:
    :return:
    '''
    delimiters = util.sentence_delimiters
    delimiters = set([util.as_text(item) for item in delimiters])
    res = [util.as_text(text)]
    util.debug(text)
    util.debug(delimiters)
    for sep in delimiters:
        text, res = res, []
        for seq in text:
            res += seq.split(sep)
    res = [s.strip() for s in res if len(s.strip()) > 0]
    return res


with open("./Data/NYTimes_Bi_Clean.json", 'r', encoding='utf-8') as fp:
    data = fp.readlines()
    # chinese_sent_num = 0
    # english_sent_num = 0
    # chinese_total_sent_len = 0
    # english_total_sent_len = 0
    # for i, line in enumerate(data):
    #     dec_json = json.loads(line)
    #     content_list = dec_json['Content']
    #     for j, d in enumerate(content_list):
    #         chinese_paragraph = d['chinese']
    #         english_paragraph = d['english']
    #         if chinese_paragraph and english_paragraph:
    #             chinese_sent_list = sent_segmentation(chinese_paragraph)
    #             english_sent_list = nltk.sent_tokenize(english_paragraph)
    #             chinese_sent_num += len(chinese_sent_list)
    #             english_sent_num += len(english_sent_list)
    #             for ch_sent in chinese_sent_list:
    #                 chinese_total_sent_len += (len(ch_sent) + 1)
    #             for en_sent in english_sent_list:
    #                 english_total_sent_len += len(nltk.word_tokenize(en_sent))
    # assert i + 1 == 18614
    # print(f"平均英文句子数量:{english_sent_num/18614}\n"
    #       f"平均中文句子数量:{chinese_sent_num/18614}\n"
    #       f"平均英文句子长度:{english_total_sent_len/english_sent_num}\n"
    #       f"平均中文句子长度:{chinese_total_sent_len/chinese_sent_num}")








    en_content_len = 0
    ch_content_len = 0
    en_summary_len = 0
    ch_summary_len = 0
    en_title_len = 0
    ch_title_len = 0
    sent_num = 0
    en_sent_len = 0
    ch_sent_len = 0
    en_content_max_len = 0
    en_summary_max_len = 0
    ch_content_max_len = 0
    ch_summary_max_len = 0
    for i,line in enumerate(data):
        dec_json = json.loads(line)
        title_dict = dec_json['Title']
        content_list = dec_json['Content']
        summary_dict = dec_json['Summary']
        english_title = title_dict['english']
        chinese_title = title_dict['chinese']
        english_summary = summary_dict['english']
        chinese_summary = summary_dict['chinese']
        en_title_len += len(nltk.word_tokenize(text=english_title))
        ch_title_len += len(chinese_title)
        en_sum_len_single = len(nltk.word_tokenize(text=english_summary))
        ch_sum_len_single = len(chinese_summary)
        en_summary_max_len = max(en_summary_max_len,en_sum_len_single)
        ch_summary_max_len = max(ch_sum_len_single,ch_summary_max_len)
        en_summary_len += en_sum_len_single
        ch_summary_len += ch_sum_len_single
        sent_num_count = 0
        ch_content_len_single = 0
        en_content_len_single = 0
        for j,d in enumerate(content_list):
            chinese_sent = d['chinese']
            english_sent = d['english']
            if  chinese_sent and  english_sent:
                ch_content_len_single += len(chinese_sent)
                en_content_len_single += len(nltk.word_tokenize(english_sent))
                sent_num_count += 1
        ch_content_len += ch_content_len_single
        en_content_len += en_content_len_single
        if en_content_len_single == 16123:
            print(english_summary)
            print(chinese_summary)
        ch_content_max_len = max(ch_content_max_len,ch_content_len_single)
        en_content_max_len = max(en_content_max_len,en_content_len_single)
        sent_num += sent_num_count
    i = 18614
    print(f"统计信息如下：\n"
          f"平均英文标题长度:{en_title_len/i}\n"
          f"平均中文标题长度:{ch_title_len/i}\n"
          f"平均英文新闻长度:{en_content_len/i}\n"
          f"平均中文新闻长度:{ch_content_len/i}\n"
          f"最大英语新闻长度:{en_content_max_len}\n"
          f"最大中文新闻长度:{ch_content_max_len}\n"
          f"平均英文摘要长度:{en_summary_len/i}\n"
          f"平均中文摘要长度:{ch_summary_len/i}\n"
          f"最大英文摘要长度:{en_summary_max_len}\n"
          f"最大中文摘要长度:{ch_summary_max_len}\n"
          f"平均句子数量:{sent_num/i}\n"
          f"平均英文句子长度:{en_content_len/sent_num}\n"
          f"平均中文句子长度:{ch_content_len/sent_num}\n")
