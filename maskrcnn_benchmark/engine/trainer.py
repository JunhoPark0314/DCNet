# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import pickle
import os
import cv2
import torch
import torch.distributed as dist
import collections
from maskrcnn_benchmark.utils.comm import get_world_size, is_main_process
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.visualize import visualize_episode
from maskrcnn_benchmark.utils.events import EventStorage, get_event_storage
from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    meta_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    phase,
    shot,
    split,
    arguments,
    cfg,
    optimizer2=None,
    writer=None,
    meta_crop_shot=1,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    
    data_iter_meta = iter(meta_loader)
    print('meta itered')
    with EventStorage(start_iter=start_iter):
        storage = get_event_storage()
        storage.max_iter = len(data_loader)
        for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            scheduler.step()

            images = images.to(device)
            try:
                meta_input, meta_info = next(data_iter_meta)
                meta_input = meta_input.to(device)
            except:
                meta_iter = iter(meta_loader)
                meta_input, meta_info = next(meta_iter)
                meta_input = meta_input.to(device)
        
            if meta_crop_shot:
                num_classes = meta_input.shape[0] // meta_crop_shot
            else:
                num_classes = meta_input.shape[0]
            
            if cfg.MODEL.SHUFFLE_CLS:
                curr_perm = torch.randperm(len(meta_input))
                reverse_map = {k.item()+1:v.item()+1 for k,v in zip(torch.arange(len(meta_input)), curr_perm)}
                meta_input = meta_input[curr_perm]
                for i, t in enumerate(targets):
                    curr_labels = t.extra_fields["labels"]
                    curr_labels = torch.tensor([reverse_map[x.item()] for x in curr_labels])
                    targets[i].extra_fields["labels"] = curr_labels

            targets = [target.to(device) for target in targets]
            loss_dict, result = model(images, targets, meta_input)
            losses = sum(loss for loss in loss_dict.values())

            assert losses.isfinite().item()

            torch.cuda.empty_cache()
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            if optimizer2 is not None:
                optimizer2.zero_grad()
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                torch.cuda.empty_cache()
                scaled_losses.backward()

            optimizer.step()

            if optimizer2 is not None:
                optimizer2.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            storage.step()

            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "max_iter: {max_iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        max_iter=max_iter,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if (iteration % 200 == 1 or iteration == max_iter) and is_main_process(): 
                visualize_episode(meta_input, meta_info, images, targets, result, writer)
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)
            
        
        #images = None
        with torch.no_grad():
            class_attentions = collections.defaultdict(list)
            meta_loader.batch_sampler.start_iter = 0
            data_loader.batch_sampler.start_iter = 0

            data_iter = iter(data_loader)
            meta_iter1 = iter(meta_loader)
            for i in range(shot):
                try:
                    meta_input = next(meta_iter1)[0]
                    #if images is None:
                    images, targets, _ = next(data_iter)
                except StopIteration:
                    meta_iter1 = iter(meta_loader)
                    meta_input = next(meta_iter1)[0]
                    #if images is None:
                    images, targets, _ = next(data_iter)

                images = images.to(device)
                meta_input = meta_input.to(device)

                if meta_crop_shot:
                    num_classes = meta_input.shape[0] // meta_crop_shot
                else:
                    num_classes = meta_input.shape[0]
            
                meta_label = []
                for n in range(num_classes):
                    meta_label.append(n)
                attentions = model(images, targets, meta_input, meta_label,average_shot=True)
                for idx in meta_label:
                    class_attentions[idx].append(attentions[idx])
        mean_class_attentions = {k: sum(v) / len(v) for k, v in class_attentions.items()}
        #mean_class_attentions = {k: mean_class_attentions[0] for k, v in class_attentions.items()}

        output_dir = os.path.join(cfg.OUTPUT_DIR,'saved_attentions/')
        save_path = os.path.join(output_dir, 'meta_type_{}'.format(split))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path,
                                str(phase) + '_shots_' + str(shot) + '_mean_class_attentions.pkl'), 'wb') as f:
            pickle.dump(mean_class_attentions, f, pickle.HIGHEST_PROTOCOL)
        print('save ' + str(shot) + ' mean classes attentions done!')                    



        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )
