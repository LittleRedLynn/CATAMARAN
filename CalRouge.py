from rouge import Rouge
import jieba
import nltk
import json

FILE_PATH = "./Data/prediction_generated_chinese_to_english_epoch5.json"
LANG = "chinese"

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    rouge = Rouge()
    data = json.load(f)
    generated_result = data["generated_result"]
    r1, r2, rl = 0, 0, 0
    count = 0
    for d in generated_result:
        gen_sum = d["generated_summary"]
        ref_sum = d["golden_summary"]
        if LANG == 'chinese':
            hyp = " ".join(jieba.cut(gen_sum))
            ref = " ".join(jieba.cut(ref_sum))
        else:
            hyp = " ".join(nltk.word_tokenize(gen_sum))
            ref = " ".join(nltk.word_tokenize(ref_sum))
        rouge_score = rouge.get_scores(hyps=hyp, refs=ref, avg=True, ignore_empty=True)
        rouge_1 = rouge_score['rouge-1']['f']
        rouge_2 = rouge_score['rouge-2']['f']
        rouge_l = rouge_score['rouge-l']['f']
        print("======================================================================================================")
        print(f"generated summary:{gen_sum}")
        print(f"reference summary:{ref_sum}")
        print(f"r1:{rouge_1},r2:{rouge_2},rl:{rouge_l}")
        r1 += rouge_1
        r2 += rouge_2
        rl += rouge_l
        count += 1
    final_r1 = r1 / count
    final_r2 = r2 / count
    final_rl = rl / count
    print(f"rouge-1:{round(final_r1 * 100.0,2)},rouge-2:{round(final_r2 * 100.0,2) },rouge-l:{round(final_rl * 100.0,2)}")
