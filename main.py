import os
import logging
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer
import json
from datetime import datetime

import config
import data_loader
import globalpoint
from utils.common_utils import set_seed, set_logger, read_json, trans_ij2k, fine_grade_tokenize
from utils.train_utils import load_model_and_parallel, build_optimizer_and_scheduler, save_model
from utils.metric_utils import calculate_metric, classification_report, get_p_r_f
from tensorboardX import SummaryWriter

args = config.Args().get_parser()
set_seed(args.seed)
logger = logging.getLogger(__name__)

if args.use_tensorboard == "True":
    writer = SummaryWriter(log_dir='./tensorboard')


class BertForNer:
    def __init__(self, args, train_loader, dev_loader, test_loader, idx2tag, label_list, model, device, dev_callback=None, test_callback=None, original_texts=None):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.args = args
        self.idx2tag = idx2tag
        self.label_list = label_list
        self.model = model
        self.device = device
        self.dev_callback = dev_callback
        self.test_callback = test_callback
        self.original_texts = original_texts
        if train_loader is not None:
            self.t_total = len(self.train_loader) * args.train_epochs
            self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        global_step = 0
        self.model.zero_grad()
        eval_steps = self.args.eval_steps
        best_f1 = 0.0
        for epoch in range(1, self.args.train_epochs+1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for batch in batch_data:
                    batch = batch.to(self.device)
                loss, logits = self.model(batch_data[0], batch_data[1], batch_data[2], batch_data[3])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{} {}/{} loss:{:.4f}'.format(epoch, global_step, self.t_total, loss.item()))

                global_step += 1
                if self.args.use_tensorboard == "True":
                    writer.add_scalar('data/loss', loss.item(), global_step)
                if global_step % eval_steps == 0:
                   dev_loss, precision, recall, f1_score = self.dev()
                   logger.info('[eval] loss:{:.4f} precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(dev_loss, precision, recall, f1_score))
                   if f1_score > best_f1:
                       save_model(self.args, self.model, model_name, global_step)
                       best_f1 = f1_score
        logger.info("best f1:{}".format(best_f1))

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            pred_entities = []
            true_entities = []
            tot_dev_loss = 0.0
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                labels = dev_batch_data[3]
                for dev_batch in dev_batch_data:
                    dev_batch = dev_batch.to(self.device)
                _, logits = self.model(dev_batch_data[0], dev_batch_data[1], dev_batch_data[2], dev_batch_data[3])
                batch_size = logits.size(0)
                dev_callbak = self.dev_callback[eval_step * batch_size:(eval_step + 1) * batch_size]

                for i in range(batch_size):
                    pred_tmp = defaultdict(list)
                    logit = logits[i, ...]
                    tokens = dev_callbak[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[j, :len(tokens), :len(tokens)]
                        for start, end in zip(*np.where(logit_.cpu().numpy() > 0.5)):
                            pred_tmp[self.idx2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    pred_entities.append(dict(pred_tmp))

                for i in range(batch_size):
                    true_tmp = defaultdict(list)
                    logit = labels[i, ...]
                    tokens = dev_callbak[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[j, :len(tokens), :len(tokens)]
                        for start, end in zip(*np.where(logit_.cpu().numpy() == 1)):
                            true_tmp[self.idx2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    true_entities.append(true_tmp)

            total_count = [0 for _ in range(len(self.idx2tag))]
            role_metric = np.zeros([len(self.idx2tag), 3])
            for pred, true in zip(pred_entities, true_entities):
                tmp_metric = np.zeros([len(self.idx2tag), 3])
                for idx, _type in enumerate(self.label_list):
                    if _type not in pred:
                        pred[_type] = []
                    total_count[idx] += len(true[_type])
                    tmp_metric[idx] += calculate_metric(true[_type], pred[_type])

                role_metric += tmp_metric

            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            return tot_dev_loss, mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]

    def test(self, model_path):
        import time
        start_time = time.time()
        logger.info('[test] start testing, model path: {}'.format(model_path))
        model = globalpoint.GlobalPointerNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        pred_entities = []
        result_entities = []
        true_entities = []
        with torch.no_grad():
            for eval_step, test_batch_data in enumerate(self.test_loader):
                labels = test_batch_data[3]
                for test_batch in test_batch_data:
                    test_batch = test_batch.to(device)
                _, logits = model(test_batch_data[0], test_batch_data[1], test_batch_data[2], test_batch_data[3])
                batch_size = logits.size(0)
                test_callback = self.test_callback[eval_step * batch_size:(eval_step + 1) * batch_size]

                for i in range(batch_size):
                    pred_tmp = defaultdict(list)
                    logit = logits[i, ...]
                    tokens = test_callback[i]
                    sample_id = eval_step * batch_size + i
                    if self.original_texts and sample_id < len(self.original_texts):
                        full_text = self.original_texts[sample_id]
                    else:
                        full_text = "".join(tokens)
                        if full_text.startswith('[CLS]'):
                            full_text = full_text[5:]
                        if full_text.endswith('[SEP]'):
                            full_text = full_text[:-5]
                    result_tmp = {
                        'full_text': full_text,
                        'entities': defaultdict(list)
                    }
                    for j in range(self.args.num_tags):
                        logit_ = logit[j, :len(tokens), :len(tokens)]
                        for start, end in zip(*np.where(logit_.cpu().numpy() > 0.5)):
                            confidence = round(float(logit_[start, end].item()), 4)
                            entity_text = "".join(tokens[start:end + 1])
                            if entity_text.startswith('[CLS]'):
                                entity_text = entity_text[5:]
                            if entity_text.endswith('[SEP]'):
                                entity_text = entity_text[:-5]
                            entity_info = [entity_text, int(start), int(end), confidence]
                            pred_tmp[self.idx2tag[j]].append(["".join(tokens[start:end + 1]), start])
                            result_tmp['entities'][self.idx2tag[j]].append(entity_info)
                    pred_entities.append(dict(pred_tmp))
                    result_entities.append(result_tmp)

                for i in range(batch_size):
                    true_tmp = defaultdict(list)
                    logit = labels[i, ...]
                    tokens = test_callback[i]
                    for j in range(self.args.num_tags):
                        logit_ = logit[j, :len(tokens), :len(tokens)]
                        for start, end in zip(*np.where(logit_.cpu().numpy() == 1)):
                            true_tmp[self.idx2tag[j]].append(["".join(tokens[start:end + 1]), start])
                    true_entities.append(true_tmp)

            total_count = [0 for _ in range(len(self.idx2tag))]
            role_metric = np.zeros([len(self.idx2tag), 3])
            for pred, true in zip(pred_entities, true_entities):
                tmp_metric = np.zeros([len(self.idx2tag), 3])
                for idx, _type in enumerate(self.label_list):
                    if _type not in pred:
                        pred[_type] = []
                    total_count[idx] += len(true[_type])
                    tmp_metric[idx] += calculate_metric(true[_type], pred[_type])

                role_metric += tmp_metric
            end_time = time.time()
            test_time = end_time - start_time
            logger.info('[test] test results:\n{}'.format(classification_report(role_metric, self.label_list, self.idx2tag, total_count, digits=4)))
            logger.info('[test] test time cost: {:.2f} seconds'.format(test_time))

            # Save confidence file - one line per sample
            os.makedirs('./json/', exist_ok=True)
            current_date = datetime.now().strftime('%Y%m%d')
            output_file = f'./json/globalpointer_{current_date}.json'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in result_entities:
                    output_item = {
                        "full_text": item['full_text'],
                        "entities": dict(item['entities'])
                    }
                    json_line = json.dumps(output_item, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            logger.info('[test] confidence file saved: {}'.format(output_file))

    def predict(self, raw_text, model_path):
        logger.info('[predict] start prediction, input text: {}'.format(raw_text))
        model = globalpoint.GlobalPointerNer(self.args)
        model, device = load_model_and_parallel(model, self.args.gpu_ids, model_path)
        model.eval()
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained(self.args.bert_dir)
            tokens = [i for i in raw_text]
            encode_dict = tokenizer.encode_plus(text=tokens,
                                                max_length=self.args.max_seq_len,
                                                padding='max_length',
                                                truncation='longest_first',
                                                is_pretokenized=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0).to(device)
            attention_masks = torch.from_numpy(np.array(encode_dict['attention_mask'], dtype=np.uint8)).unsqueeze(0).to(device)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(device)
            logits = model(token_ids, attention_masks, token_type_ids, None)
            batch_size = logits.size(0)
            pred_tmp = defaultdict(list)
            for i in range(batch_size):
              logit = logits[i, ...]
              for j in range(self.args.num_tags):
                  logit_ = logit[j, :len(tokens), :len(tokens)]
                  for start, end in zip(*np.where(logit_.cpu().numpy() > 0.5)):
                      pred_tmp[self.idx2tag[j]].append(["".join(tokens[start:end + 1]), start-1])

            logger.info('[predict] prediction result: {}'.format(dict(pred_tmp)))


if __name__ == '__main__':
    data_name = 'c'
    if args.use_efficient_globalpointer == "True":
      model_name = 'bert-1-eff'
    else:
      model_name = 'bert-1'
    set_logger(os.path.join(args.log_dir, '{}.log'.format(model_name)))
    if data_name == "c":
        args.data_dir = './data/CMeEE'
        data_path = os.path.join(args.data_dir, 'mid_data')
        label_list = read_json(data_path, 'labels')
        tag2id = {}
        id2tag = {}
        for k, v in enumerate(label_list):
            tag2id[v] = k
            id2tag[k] = v

        logger.info(args)
        max_seq_len = args.max_seq_len
        tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)

        model = globalpoint.GlobalPointerNer(args)
        model, device = load_model_and_parallel(model, args.gpu_ids)


        collate = data_loader.Collate(max_len=max_seq_len, tag2id=tag2id, device=device)

        train_dataset, _, _ = data_loader.MyDataset(file_path=os.path.join(data_path, 'train.json'),
                                                 tokenizer=tokenizer,
                                                 max_len=max_seq_len)
        print(train_dataset[0])
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=collate.collate_fn)
        dev_dataset, dev_callback, _ = data_loader.MyDataset(file_path=os.path.join(data_path, 'dev.json'),
                                                          tokenizer=tokenizer,
                                                          max_len=max_seq_len)
        print(dev_dataset[0])
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                collate_fn=collate.collate_fn)
        test_dataset, test_callback, original_texts = data_loader.MyDataset(file_path=os.path.join(data_path, 'test.json'),
                                                           tokenizer=tokenizer,
                                                           max_len=max_seq_len)
        print(test_dataset[0])
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=collate.collate_fn)

        bertForNer = BertForNer(args, train_loader, dev_loader, test_loader, id2tag, label_list, model, device, dev_callback, test_callback, original_texts)

        if args.only_test:
            model_path = os.path.join(args.output_dir, model_name, 'model.pt')
            bertForNer.test(model_path)
        else:
            bertForNer.train()
            model_path = os.path.join(args.output_dir, model_name, 'model.pt')
            bertForNer.test(model_path)

            raw_text = "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。"  # Chinese test sample (kept as data)
            logger.info(raw_text)
            bertForNer.predict(raw_text, model_path)