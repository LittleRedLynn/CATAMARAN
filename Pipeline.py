import re
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict
import os
from pathlib import Path
from torch.utils.data import dataset, dataloader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import MBartForConditionalGeneration, MBartTokenizer


class HyperParameter:

    def __init__(
            self,
            batch_size: int,
            src_lang: str,
            tgt_lang: str,
            max_src_len: int,
            max_tgt_len: int,
            epochs: int,
            warm_up_step: int,
            lr: float,
            device: torch.device,
            patience: int,
            checkpoints_path: str,
            beam_size: int
    ):
        """
        超参数类
        """
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.patience = patience
        self.checkpoints_path = checkpoints_path
        self.beam_size = beam_size
        self.warm_up_step = warm_up_step
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang


class MyDataset(dataset.Dataset):

    def __init__(
            self,
            path: str,
            args: HyperParameter
    ):
        super(MyDataset, self).__init__()
        self.args = args
        self.lang_map = dict(chinese='zh_CN', english='en_XX')
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25",
                                                        src_lang=self.lang_map[self.args.src_lang],
                                                        tgt_lang=self.lang_map[self.args.tgt_lang])
        try:
            fp = open(path, 'r', encoding='utf-8')
        except (FileNotFoundError, FileExistsError) as e:
            print(f"通过文件路径{path}打开文件失败，错误信息为{e}")
            exit(1)
        else:
            self.data = fp.readlines()
            fp.close()
        self.dec_data = [json.loads(line) for line in self.data]

    def __getitem__(self, index):
        dec_json = self.dec_data[index]
        if self.args.src_lang == "chinese":
            src = dec_json['chinese_content']
        elif self.args.src_lang == "english":
            src = dec_json['english_content']
        else:
            raise KeyError("超参数\'src_lang\'设置错误，请检查")

        if self.args.tgt_lang == "chinese":
            tgt = dec_json['chinese_summary']
        elif self.args.tgt_lang == "english":
            tgt = dec_json['english_summary']
        else:
            raise KeyError("超参数\'tgt_lang\'设置错误，请检查")
        inputs = self.tokenizer(text=src,
                                padding='max_length',
                                truncation=True,
                                max_length=self.args.max_src_len,
                                return_tensors='pt')
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(text=tgt,
                                    padding="max_length",
                                    max_length=self.args.max_tgt_len,
                                    truncation=True,
                                    return_tensors='pt').input_ids
        decoder_input_string = self.lang_map[self.args.tgt_lang] + " " + tgt + " " + "</s>"
        decoder_input = self.tokenizer(text=decoder_input_string,
                                       add_special_tokens=False,
                                       padding='max_length',
                                       max_length=self.args.max_tgt_len,
                                       truncation=True,
                                       return_tensors='pt').input_ids
        # decoder_input = decoder_input.to(self.args.device)
        # labels = labels.to(self.args.device)
        return input_ids.squeeze(), attention_mask.squeeze(), decoder_input.squeeze(), labels.squeeze()

    def __len__(self):
        return len(self.data)


class MyDataloader:

    def __init__(
            self,
            dataset_dict: dict,
            args: HyperParameter,
            num_workers: int,
    ):
        """
        :param dataset_dict: 初始化Dataloader用的包含Dataset的字典，格式为{"split":MyDataset}，
        split代表这个数据集的分割名(train,eval,pred)，value是上面定义的MyDataset类
        :param args: 超参数类
        :param num_workers: 同torch.utils.data.dataloader.DataLoader
        """
        self.dataloader_dict = dict()
        if not dataset_dict.keys():
            raise KeyError("参数\'dataset_dict\'不能为空!")
        for k, v in dataset_dict.items():
            shuffle = True if k == 'train' else False
            self.dataloader_dict[k] = dataloader.DataLoader(
                dataset=v,
                batch_size=args.batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                drop_last=True
            )
        self.split = list(dataset_dict.keys())[0]

    def set_split(self, split: str):
        assert split in self.dataloader_dict, \
            f"split设置错误，必须为{self.dataloader_dict.keys()}中的一种"
        self.split = split

    def get_dataloader(self):
        return self.dataloader_dict[self.split]

    @classmethod
    def from_path_dict(
            cls,
            path_dict: dict,
            args: HyperParameter,
            num_workers: int,
    ):
        """
        类初始化方法
        :param path_dict: 以split为键, file_path为值的字典
        :param args: 同构造方法
        :param num_workers: 同构造方法
        """
        dataset_dict = dict()
        for k, v in path_dict.items():
            dataset_dict[k] = MyDataset(v, args)
        return cls(dataset_dict, args, num_workers)

    def get_num_batches(self):
        return len(self.dataloader_dict[self.split])


class Pipeline:

    def __init__(
            self,
            path_dict: dict,
            num_workers: int,
            args: HyperParameter,
            model_checkpoint=None
    ):
        self.args = args
        self.lang_map = dict(chinese='zh_CN', english='en_XX')
        self.dataloader = MyDataloader.from_path_dict(
            path_dict=path_dict,
            args=args,
            num_workers=num_workers,
        )
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25",
                                                        src_lang=self.lang_map[self.args.src_lang],
                                                        tgt_lang=self.lang_map[self.args.tgt_lang])
        if model_checkpoint:
            self.model = MBartForConditionalGeneration.from_pretrained(model_checkpoint)
        else:
            self.model = MBartForConditionalGeneration.from_pretrained("./Pretrained")
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = AdamW(params=self.model.parameters(), lr=args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=args.warm_up_step,
                                                         num_training_steps=
                                                         args.epochs * self.dataloader.get_num_batches())
        self.training_state = dict(
            training_loss=[],
            validation_loss=[],
            prediction_loss=[]
        )
        self.model.to(args.device)

    def is_early_stop(self):
        if len(self.training_state["training_loss"]) <= self.args.warm_up_step // 10: return False
        valid_window = self.training_state["training_loss"][-1:-self.args.patience - 1:-1]
        all([e_loss > valid_window[index + 1] for index, e_loss in enumerate(valid_window[:-1])])

    def train(self):
        cur_step = 1
        early_stop_flag = False
        for epoch_index in range(self.args.epochs):
            self.model.train()
            self.dataloader.set_split("train")
            bar = tqdm(enumerate(self.dataloader.get_dataloader()))
            bar.set_description(f"Training Epoch {epoch_index} ", refresh=True)
            running_loss = 0.0
            for batch_index, batch in bar:
                self.optimizer.zero_grad()
                input_ids, attention_mask, decoder_input, labels = batch
                input_ids, attention_mask, decoder_input, labels = input_ids.to(self.args.device), \
                                                                   attention_mask.to(self.args.device), \
                                                                   decoder_input.to(self.args.device), \
                                                                   labels.to(self.args.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     decoder_input_ids=decoder_input,
                                     labels=labels
                                     )
                loss = outputs.loss
                loss = loss.mean()
                # loss = self.loss_func(input=outputs.logits.permute(0, 2, 1).contiguous(), target=labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                raw_loss = loss.clone().detach().mean().item()
                running_loss += (raw_loss - running_loss) / (batch_index + 1)
                bar.set_postfix(loss=running_loss)
                if cur_step % 100 == 0:
                    self.training_state["training_loss"].append(running_loss)
                    if self.is_early_stop():
                        print("early stop triggered! training loop finished.")
                        early_stop_flag = True
                        break
                cur_step += 1
            if early_stop_flag: break
            eval_loss = self.val_or_pred("validation")
            self.training_state["validation_loss"].append(eval_loss)
            save_dir = f"{self.args.src_lang}2{self.args.tgt_lang}_epoch{epoch_index}_checkpoints"
            file_path = Path(f"./{save_dir}")
            if not file_path.exists():
                os.system(f"mkdir {save_dir}")
            try:
                if torch.cuda.device_count() > 1:
                    self.model.module.save_pretrained(f"./{save_dir}")
                else:
                    self.model.save_pretrained(f"./{save_dir}")
            except Exception as e:
                print("模型保存出错，错误信息{}".format(e))
            else:
                print(f"模型 checkpoint {cur_step} 保存成功!")

        pred_loss = self.val_or_pred("prediction")
        self.training_state["prediction_loss"].append(pred_loss)
        print(f"train loop terminated, final pred_loss: {pred_loss}")
        return self.training_state

    def val_or_pred(self, split: str):
        self.model.eval()
        running_loss = 0.0
        bar = tqdm(enumerate(self.dataloader.dataloader_dict[split]))
        bar.set_description(f"{split} processing")
        with torch.no_grad():
            for batch_index, batch in bar:
                self.optimizer.zero_grad()
                input_ids, attention_mask, decoder_input, labels = batch
                input_ids, attention_mask, decoder_input, labels = input_ids.to(self.args.device), \
                                                                   attention_mask.to(self.args.device), \
                                                                   decoder_input.to(self.args.device), \
                                                                   labels.to(self.args.device)
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     decoder_input_ids=decoder_input,
                                     labels=labels
                                     )
                loss = outputs.loss
                loss = loss.mean()
                # loss = self.loss_func(input=outputs.logits.permute(0, 2, 1).contiguous(), target=labels)
                raw_loss = loss.clone().detach().mean().item()
                running_loss += (raw_loss - running_loss) / (batch_index + 1)
                bar.set_postfix(loss=running_loss)
        print(f"{split} progress finished. average loss is {running_loss}")
        return running_loss

    def generate(self):
        self.model.eval()
        assert self.args.batch_size == 1, "generate时batch_size必须设为1!"
        bar = tqdm(enumerate(self.dataloader.dataloader_dict['prediction']))
        res = []
        with torch.no_grad(), open(f"./Data/prediction_generated_{self.args.src_lang}_to_{self.args.tgt_lang}.json",
                                   'w', encoding='utf-8') as fp:
            for batch_index, batch in bar:
                bar.set_description(f"generate step {batch_index + 1} ")
                input_ids, attention_mask, decoder_input, labels = batch
                input_ids, attention_mask, decoder_input, labels = input_ids.to(self.args.device), \
                                                                   attention_mask.to(self.args.device), \
                                                                   decoder_input.to(self.args.device), \
                                                                   labels.to(self.args.device)
                generated_tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                       decoder_start_token_id=self.tokenizer.lang_code_to_id[
                                                           self.lang_map[self.args.tgt_lang]],
                                                       num_beams=self.args.beam_size,
                                                       max_length=self.args.max_tgt_len, early_stopping=True)

                print(f"generated_tokens:{generated_tokens}")
                generated_summary = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                print(f"generated_summary:{generated_summary}")
                print("-----------------------------------------------------------------------------------------------")
                original_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
                golden_summary = self.tokenizer.batch_decode(labels, skip_special_tokens=True)[0]
                res.append(dict(generated_summary=generated_summary, golden_summary=golden_summary,
                                original_text=original_text))
            enc_json = json.dumps({"generated_result": res}, ensure_ascii=False, indent=4)
            print(enc_json, file=fp)
        print(f"测试集所有摘要生成完毕，结果保存在 \'./Data/prediction_generated_{self.args.src_lang}_to_{self.args.tgt_lang}.json\' 中")
        return

    def decode(self, src_text: str):
        """
        return: a summary in a different language from the src text
        """
        with torch.no_grad():
            inputs = self.tokenizer(text=src_text,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.args.max_src_len,
                                    return_tensors='pt')
            input_ids = inputs.input_ids.to(self.args.device)
            attention_mask = inputs.attention_mask.to(self.args.device)
            generated_tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                   max_length=self.args.max_tgt_len,
                                                   num_beams=self.args.beam_size,
                                                   decoder_start_token_id=
                                                   self.tokenizer.lang_code_to_id[
                                                       self.lang_map[self.args.tgt_lang]])
            return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def main(hyp: HyperParameter,
         path: dict,
         model_checkpoint=None,
         operation='train',
         text=None):
    pipeline = Pipeline(path_dict=path, num_workers=4,
                        args=hyp, model_checkpoint=model_checkpoint)
    assert operation in ['train', 'generate', 'inference'], "只能执行train,generate,inference三种操作，请检查命令行输入"
    if operation == 'train':
        state = pipeline.train()
        print("训练完毕,训练记录信息如下:")
        training_loss = [state['training_loss'][0], state['training_loss'][-1]]
        print(f"training_loss:{training_loss}\n"
              f"validation_loss:{state['validation_loss']}\n"
              f"prediction_loss:{state['prediction_loss']}")
    elif operation == 'generate':
        pipeline.generate()
    else:
        if not text:
            print("请输入正确的新闻文本!")
            return
        summary = pipeline.decode(src_text=text)
        print(f"摘要为:{summary}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", type=int, default=4, help="批大小，默认为4")
    parser.add_argument("-src_lang", type=str, required=True, choices=['chinese', 'english'],
                        help="原文的语言，\'chinese\'或者\'english\'")
    parser.add_argument("-tgt_lang", type=str, required=True, choices=['chinese', 'english'],
                        help="摘要的语言，\'chinese\'或者\'english\'")
    parser.add_argument("-max_src_len", type=int, default=1024, help="最大输入长度")
    parser.add_argument("-max_tgt_len", type=int, default=128, help="最大输出长度")
    parser.add_argument("-epochs", type=int, required=True, help="总的训练迭代次数")
    parser.add_argument("-warm_up_step", type=int, required=True, help="升温步数")
    parser.add_argument("-lr", type=float, required=True, help="学习率")
    parser.add_argument("-device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("-patience", type=int, default=6, help="早停的忍耐度")
    parser.add_argument("-checkpoints_path", type=str, required=True, help="保存模型的路径")
    parser.add_argument("-beam_size", type=int, default=3, help="集束搜索的光束个数")
    parser.add_argument("-path_dict", type=str, help="训练、验证和预测数据集的路径字典")
    parser.add_argument("-model_checkpoint", default=None, help="训练好的模型路径")
    parser.add_argument("-operation", type=str, choices=['train', 'generate', 'inference'],
                        help="train:训练,generate:对测试集进行生成,inference:对输入的一条原文进行摘要")
    parser.add_argument("-text", type=str, default=None, help="进行推断的摘要")
    opt = parser.parse_args()
    args = HyperParameter(
        batch_size=opt.batch_size,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        max_src_len=opt.max_src_len,
        max_tgt_len=opt.max_tgt_len,
        epochs=opt.epochs,
        warm_up_step=opt.warm_up_step,
        lr=opt.lr,
        device=opt.device,
        patience=opt.patience,
        checkpoints_path=opt.checkpoints_path,
        beam_size=opt.beam_size
    )
    path_dict = eval(opt.path_dict)
    checkpoint = opt.model_checkpoint
    operation = opt.operation
    text = opt.text
    main(args, path_dict, checkpoint, operation, text)
