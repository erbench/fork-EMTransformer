import logging
import os

import torch
from tqdm import tqdm, trange

from logging_customized import setup_logging
from model import save_model
from tensorboardX import SummaryWriter
import time

setup_logging()


def train(device,
          train_dataloader,
          model,
          optimizer,
          scheduler,
          evaluation,
          num_epocs,
          max_grad_norm,
          save_model_after_epoch,
          experiment_name,
          output_dir,
          model_type,
          testing=None):
    logging.info("***** Run training *****")
    tb_writer = SummaryWriter(os.path.join(output_dir, experiment_name))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # we are interested in 0 shot learning, therefore we already evaluate before training.
    eval_results = evaluation.evaluate(model, device, -1)
    for key, value in eval_results.items():
        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
    results = []
    best_epoch = []
    best_f1 = 0
    t_start = time.process_time()
    for epoch in trange(int(num_epocs), desc="Epoch"):
        t_epoch = time.process_time()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}

            if model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            global_step += 1

            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss), global_step)
            logging_loss = tr_loss
        t_train = time.process_time()
        eval_results = evaluation.evaluate(model, device, epoch)
        f1 = eval_results['f1']
        t_eval = time.process_time()
        if type(testing) != type(None):
            eval_results = testing.evaluate(model, device, epoch)
        t_test = time.process_time()
        curr_results = [epoch, eval_results['f1'], eval_results['precision'], eval_results['recall'], t_train - t_epoch, t_eval - t_train, t_test - t_eval]
        results += [curr_results]

        if f1 > best_f1:
            best_epoch = curr_results
            best_f1 = f1

        for key, value in eval_results.items():
            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

        if save_model_after_epoch:
            save_model(model, experiment_name, output_dir, epoch=epoch)

        if (t_test - t_start) + (t_test - t_epoch) > 8 * 60 * 60:  # if running the next epoch would lead to a total runtime
            # higher than 8 hours, break.
            break
    results += [best_epoch]

    tb_writer.close()
    return results
