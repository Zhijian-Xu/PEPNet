import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score, f1_score
import numpy as np
from tqdm import tqdm
from utils.config import *

class Mcc_Metric:
    def __init__(self, mcc = 0.):
        if type(mcc).__name__ == 'dict':
            self.mcc = mcc['mcc']
        elif type(mcc).__name__ == 'Mcc_Metric':
            self.mcc = mcc.mcc
        else:
            self.mcc = mcc

    def better_than(self, other):
        if self.mcc > other.mcc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['mcc'] = self.mcc
        return _dict

class Loss_Metric:
    def __init__(self, loss=float('inf')):
        if type(loss).__name__ == 'dict':
            self.loss = loss['loss']
        elif type(loss).__name__ == 'Loss_Metric':
            self.loss = loss.loss
        else:
            self.loss = loss

    def better_than(self, other):
        return self.loss < other.loss

    def state_dict(self):
        return {'loss': self.loss}
    
# --- DDP Cleanup ---
def cleanup():
    """Destroys the distributed environment."""
    dist.destroy_process_group()
def filter_state_dict(model: torch.nn.Module, state_dict: dict):
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_dict[k] = v
        else:
            print(f"Skipped loading parameter: {k} | "
                  f"checkpoint shape: {v.shape} != model shape: {model_dict.get(k, 'N/A').shape}")
    return filtered_dict
# --- Main Training/Validation Function ---
def run_net(args, config, train_writer=None, val_writer=None):
    rank = args.local_rank
    world_size = args.world_size
    """Main function for training and validation using DDP."""
    is_main_process = (rank == 0) # Check if it's the main process

    if is_main_process:
        logger = get_logger(args.log_name) # Log only on main process
            # log 
        log_args_to_file(args, 'args', logger = logger)
        log_config_to_file(config, 'config', logger = logger)
    # Build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)

    # Build model
    base_model = builder.model_builder(config.model)
    base_model.to(rank) # Move model to the correct device (GPU rank)
    # Parameter setting
    start_epoch = 0
    best_metrics = Mcc_Metric(0.)
    best_metric_pretrain = Loss_Metric(float('inf')) # For pretraining loss
    metrics = Mcc_Metric(0.)
    metrics_pretrain = Loss_Metric(float('inf')) # For pretraining loss
    # Resume ckpts
    if args.resume:
        # Load checkpoint before wrapping with DDP
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger if is_main_process else None)
        best_metrics = Mcc_Metric(best_metric) # Corrected variable name
    else:
        if args.ckpts is not None:
            if is_main_process:
                print_log(f"Loading encoder weights from pre-trained checkpoint: {args.ckpts}", logger=logger)
            ckpt = torch.load(args.ckpts, map_location='cpu')
            pretrain_state_dict = ckpt.get('base_model', ckpt)

            encoder_state_dict = {}
            for k, v in pretrain_state_dict.items():
                if k.startswith('shared_encoder.'):
                    encoder_state_dict[k] = v

            filtered_state_dict = filter_state_dict(base_model, encoder_state_dict)

            incompatible_keys = base_model.load_state_dict(filtered_state_dict, strict=False)

            if is_main_process:


                print_log("\nKeys not found in the new model (should be empty or pretrain-specific):", logger=logger)
                for k in incompatible_keys.missing_keys:
                    print_log(f"  - {k}", logger=logger)

                print_log("\nKeys in the new model not found in the checkpoint (should be decoder, cls_head, etc.):", logger=logger)
                for k in incompatible_keys.unexpected_keys:
                    print_log(f"  - {k}", logger=logger)
        else:
            if is_main_process:
                print_log('Training from scratch', logger=logger)

    # DDP Wrapping
    if args.distributed:
        base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
        if is_main_process:
            print_log('Using Synchronized BatchNorm ...', logger=logger)
    # Wrap model with DDP
        base_model = DDP(base_model, device_ids=[rank], find_unused_parameters=True) # Added find_unused_parameters
        if is_main_process:
            print_log('Using Distributed Data parallel ...', logger=logger)
    elif args.use_gpu: # Handle single GPU case
         base_model.to('cuda') # Move to default cuda device if not distributed
         if is_main_process:
             print_log('Using single GPU ...', logger=logger)
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        # Load optimizer state after potentially wrapping model
        builder.resume_optimizer(optimizer, args, logger=logger if is_main_process else None)

    # Trainval loop
    for epoch in tqdm(range(start_epoch, config.max_epoch + 1), total=config.max_epoch, desc="Training: ", unit="epoch"):
        # --- SET EPOCH for the distributed sampler ---
        if args.distributed and train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
            if epoch == 0 and is_main_process:
                print_log("[DDP] Setting epoch for sampler", logger=logger)
        if is_main_process:
            tqdm.write(f"Starting Epoch {epoch}/{config.max_epoch}")

        base_model.train()

        epoch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # Use a simple list for losses, aggregation happens later
        if args.finetune_model:
            epoch_losses = {'rotation': [], 'focal': [], 'bce': [], 'dice': []}
        else:
            epoch_losses = {'Loss': [], 'L_aa': [], 'L_sol': [], 'L_pssm': []}
        total_valid_residues_epoch = 0

        batch_start_time = time.time()
        # Use tqdm only on the main process for cleaner output
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),  desc=f"Epoch {epoch} Training", unit="batch", disable=not is_main_process)

        for idx, batch in enumerate(train_iterator):
            # Move batch data to the correct device
            # Assuming batch is a dictionary or structure where values are tensors
            if isinstance(batch, dict):
                 batch = {k: v.to(rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                 batch = [item.to(rank) if isinstance(item, torch.Tensor) else item for item in batch]
            else:
                 # Handle other potential batch types if necessary
                 pass # Or raise error if type is unexpected

            data_time.update(time.time() - batch_start_time)

            # Forward pass
            # DDP handles gradient synchronization automatically
            if args.finetune_model:
                valid_probs, loss_rotation, valid_epitopes, loss_focal, loss_bce, loss_dice = base_model(batch)

                if loss_bce is None or torch.isnan(loss_bce) or torch.isinf(loss_bce):
                    print_log(f"WARNING: Invalid loss detected at epoch {epoch}, batch {idx}. Skipping batch.", logger=logger if is_main_process else None)
                    optimizer.zero_grad() # Clear potentially bad gradients
                    continue # Skip backpropagation for this batch

                valid_residues = valid_epitopes.shape[0] if valid_epitopes is not None else 0 # Handle potential None

                # Accumulate loss and count for averaging later

                # Accumulate all losses
                epoch_losses['rotation'].append(loss_rotation.item())
                epoch_losses['focal'].append(loss_focal.item() * valid_residues)
                epoch_losses['bce'].append(loss_bce.item() * valid_residues)
                epoch_losses['dice'].append(loss_dice.item() * valid_residues)
                total_valid_residues_epoch += valid_residues
                combined_loss = loss_bce 
            else: 
                loss, L_aa, L_sol, L_cent = base_model(batch)
                epoch_losses['Loss'].append(loss.item())
                epoch_losses['L_aa'].append(L_aa.item())
                epoch_losses['L_sol'].append(L_sol.item())
                epoch_losses['L_pssm'].append(L_cent.item())
                combined_loss = loss

            # Backward pass
            optimizer.zero_grad()
            combined_loss.backward() 

            # Gradient clipping (optional)
            if config.get('grad_norm_clip') is not None:
                # Clip gradients for the underlying model parameters
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2) # ???

            optimizer.step()

            # Timing and Logging (mostly on main process)
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if is_main_process and train_writer is not None and idx % 10 == 0: # Log batch loss less frequently
                n_itr = epoch * len(train_dataloader) + idx
                # Log instantaneous loss from rank 0 (or reduced loss if needed)
                if args.finetune_model:
                    train_writer.add_scalar('Train/Batch/Loss_Rotation', loss_rotation.item(), n_itr)
                    train_writer.add_scalar('Train/Batch/Loss_Focal', loss_focal.item(), n_itr)
                    train_writer.add_scalar('Train/Batch/Loss_BCE', loss_bce.item(), n_itr)
                    train_writer.add_scalar('Train/Batch/Loss_Dice', loss_dice.item(), n_itr)
                    train_iterator.set_postfix(
                    rotation_loss=loss_rotation.item(),
                    focal_loss=loss_focal.item(),
                    bce_loss=loss_bce.item(),
                    dice_loss=loss_dice.item(),
                    lr=optimizer.param_groups[0]['lr']
                )
                else:
                    train_writer.add_scalar('Train/Batch/Loss', loss.item(), n_itr)
                    train_writer.add_scalar('Train/Batch/L_AA', L_aa.item(), n_itr)
                    train_writer.add_scalar('Train/Batch/L_Sol', L_sol.item(), n_itr)
                    train_writer.add_scalar('Train/Batch/L_Pssm', L_cent.item(), n_itr)
                    train_iterator.set_postfix(
                        loss=loss.item(),
                        L_aa=L_aa.item(),
                        L_sol=L_sol.item(),
                        L_pssm=L_cent.item(),
                        lr=optimizer.param_groups[0]['lr']
                    )
                train_writer.add_scalar('Train/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
                


        # --- End of Epoch ---
        # Aggregate losses across all processes
        if args.distributed:
            for loss_type in epoch_losses:
                total_loss_tensor = torch.tensor(sum(epoch_losses[loss_type]), device=rank)
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                epoch_losses[loss_type] = total_loss_tensor.item()

            total_residues_tensor = torch.tensor(total_valid_residues_epoch, device=rank)
            dist.all_reduce(total_residues_tensor, op=dist.ReduceOp.SUM)
            total_valid_residues_epoch = total_residues_tensor.item()
        else:
            for loss_type in epoch_losses:
                epoch_losses[loss_type] = sum(epoch_losses[loss_type])
        # Calculate average losses
        avg_losses = {}
        for loss_type in epoch_losses:
            if loss_type != 'rotation' and args.finetune_model:
                avg_losses[loss_type] = epoch_losses[loss_type] / total_valid_residues_epoch if total_valid_residues_epoch > 0 else 0.0
            else:
                # print( "==============>", len(train_dataloader))
                avg_losses[loss_type] = epoch_losses[loss_type] / (len(train_dataloader)*args.world_size) if len(train_dataloader) > 0 else 0.0


        # Scheduler Step
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        epoch_end_time = time.time()

        # Logging Epoch Results (only on main process)
        if is_main_process:
            print_log("--------------------------------------------------------------------------------------------", logger=logger)
            if args.finetune_model:
                print_log(f"Train/Epoch_{epoch}/Loss_Rotation: {avg_losses['rotation']:.4f}", logger=logger)
                print_log(f"Train/Epoch_{epoch}/Loss_Focal: {avg_losses['focal']:.4f}", logger=logger)
                print_log(f"Train/Epoch_{epoch}/Loss_BCE: {avg_losses['bce']:.4f}", logger=logger)
                print_log(f"Train/Epoch_{epoch}/Loss_Dice: {avg_losses['dice']:.4f}", logger=logger)
            else:
                print_log(f"Train/Epoch_{epoch}/loss: {avg_losses['Loss']:.4f}", logger=logger)
                print_log(f"Train/Epoch_{epoch}/L_aa {avg_losses['L_aa']:.4f}", logger=logger)
                print_log(f"Train/Epoch_{epoch}/L_Sol: {avg_losses['L_sol']:.4f}", logger=logger)
                print_log(f"Train/Epoch_{epoch}/L_Pssm: {avg_losses['L_pssm']:.4f}", logger=logger)
            print_log(f"Train/Epoch_{epoch}/LR: {optimizer.param_groups[0]['lr']:.6f}", logger=logger)
            print_log(f"Epoch {epoch} Time: {epoch_end_time - epoch_start_time:.2f}s", logger=logger)
            print_log("--------------------------------------------------------------------------------------------", logger=logger)
            if train_writer is not None:
                if args.finetune_model:
                    train_writer.add_scalar('Train/Epoch/Loss_Rotation', avg_losses['rotation'], epoch)
                    train_writer.add_scalar('Train/Epoch/Loss_Focal', avg_losses['focal'], epoch)
                    train_writer.add_scalar('Train/Epoch/Loss_BCE', avg_losses['bce'], epoch)
                    train_writer.add_scalar('Train/Epoch/Loss_Dice', avg_losses['dice'], epoch)
                else:
                    train_writer.add_scalar('Train/Epoch/Loss', avg_losses['Loss'], epoch)
                    train_writer.add_scalar('Train/Epoch/L_AA', avg_losses['L_aa'], epoch)
                    train_writer.add_scalar('Train/Epoch/L_Sol', avg_losses['L_sol'], epoch)
                    train_writer.add_scalar('Train/Epoch/L_Pssm', avg_losses['L_pssm'], epoch)
                train_writer.add_scalar('Train/Epoch/LR', optimizer.param_groups[0]['lr'], epoch)
                    

        # Validation Step (run on all processes, but aggregate and save/log on main process)
        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate(base_model, test_dataloader, epoch, val_writer if is_main_process else None, rank, world_size, args, config, logger=logger if is_main_process else None)

            # Ensure all processes reach here before rank 0 checks metrics and saves
            if args.distributed:
                dist.barrier()

            if is_main_process:
                if args.finetune_model:
                    better = metrics.better_than(best_metrics)
                    # Save checkpoints only on the main process
                    if better:
                        best_metrics = metrics
                        print_log(f"*** New Best MCC: {best_metrics.mcc:.4f} at Epoch {epoch} ***", logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                        print_log("--------------------------------------------------------------------------------------------", logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
                else:
                    better = metrics.better_than(best_metric_pretrain)
                    if better:
                        best_metric_pretrain = metrics
                        print_log(f"*** New Best Pretrain Loss: {best_metric_pretrain.loss:.4f} at Epoch {epoch} ***", logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metric_pretrain, 'ckpt-pretrain-best', args, logger=logger)
                        print_log("--------------------------------------------------------------------------------------------", logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metric_pretrain, 'ckpt-last', args, logger=logger)
                
        # Ensure all processes finish the epoch before starting the next one
        if args.distributed:
            dist.barrier()


    # --- End of Training ---
    if is_main_process:
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()
        print_log("Training finished.", logger=logger)

    if args.distributed:
        cleanup()


# --- Validation Function ---
def validate(base_model, test_dataloader, epoch, val_writer, rank, world_size, args, config, logger=None):
    """Validation function compatible with DDP."""
    is_main_process = (rank == 0)
    if is_main_process:
        print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)

    base_model.eval()  # Set model to eval mode

    all_pred_probs = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        # Use tqdm only on the main process
        test_iterator = tqdm(test_dataloader, desc=f"Epoch {epoch} Validation", total=len(test_dataloader), unit="batch", disable=not is_main_process)
        for idx, batch in enumerate(test_iterator):
             # Move batch data to the correct device
            if isinstance(batch, dict):
                 batch = {k: v.to(rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                 batch = [item.to(rank) if isinstance(item, torch.Tensor) else item for item in batch]
            else:
                 pass # Handle other types
            if args.finetune_model:
                valid_probs, _, valid_epitopes,_,_,_ = base_model(batch) # Get logits and labels

                # Store results from this process
                if valid_probs is not None and valid_epitopes is not None:
                    all_pred_probs.append(valid_probs.cpu())
                    all_labels.append(valid_epitopes.cpu())
            else:
                loss, _,_,_ = base_model(batch) # For pretraining, we might not have valid_probs
                total_loss += loss.item()
                total_count += 1

    if not args.finetune_model:
        if args.distributed:
            dist.barrier()
            total_loss_tensor = torch.tensor(total_loss, device=rank)
            total_count_tensor = torch.tensor(total_count, device=rank)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count_tensor, op=dist.ReduceOp.SUM)
            avg_loss = total_loss_tensor.item() / max(total_count_tensor.item(), 1)
        else:
            avg_loss = total_loss / max(total_count, 1)
        if is_main_process:
            print_log(f"[VALIDATION] Pretrain avg loss: {avg_loss:.4f}", logger=logger)
            # Add testing results to TensorBoard
            if val_writer is not None:
                val_writer.add_scalar('Val/Epoch/loss', avg_loss, epoch)
        if args.distributed:
            dist.barrier()
        return Loss_Metric(avg_loss)# Return loss for pretraining
    
    # --- Gather results from all processes ---
    if args.distributed:
        # Synchronize before gathering
        dist.barrier()

        # Gather lists of tensors from all processes
        gathered_probs_lists = [None] * world_size
        gathered_labels_lists = [None] * world_size
        dist.all_gather_object(gathered_probs_lists, all_pred_probs)
        dist.all_gather_object(gathered_labels_lists, all_labels)

        # Concatenate results on the main process
        if is_main_process:
            # Flatten the lists of lists and concatenate tensors
            flat_probs = [tensor for sublist in gathered_probs_lists for tensor in sublist]
            flat_labels = [tensor for sublist in gathered_labels_lists for tensor in sublist]
            if flat_probs: # Check if list is not empty
                 test_pred = torch.cat(flat_probs, dim=0)
                 test_label = torch.cat(flat_labels, dim=0)
            else: # Handle case where no valid data was processed
                 test_pred = torch.empty((0,))
                 test_label = torch.empty((0,))
    else:
        # Concatenate results for single GPU/CPU
        if all_pred_probs:
            test_pred = torch.cat(all_pred_probs, dim=0)
            test_label = torch.cat(all_labels, dim=0)
        else:
            test_pred = torch.empty((0,))
            test_label = torch.empty((0,))


    # --- Calculate metrics only on the main process ---
    mcc, precision, recall, f1, aucroc = 0.0, 0.0, 0.0, 0.0, 0.0 # Default values
    final_metrics = Mcc_Metric(0.)

    if is_main_process and test_label.numel() > 0: # Ensure there are labels to evaluate
        test_label_np = test_label.numpy()

        # --- Calculate Metrics ---
        test_pred_prob_np = test_pred.numpy()

        print_log(f"Validate==>Epoch {epoch} "
                  f"pred_prob mean={test_pred.mean().item():.4f}, "
                  f"pred_prob std={test_pred.std().item():.4f}", logger=logger)

        best_threshold = 0.5
        best_mcc = -1  

        # 遍历多个阈值
        thresholds = np.arange(0.0, 1.01, 0.01)
        for threshold in thresholds:
            test_pred_binary = (test_pred_prob_np > threshold).astype(int)
            try:
                mcc = matthews_corrcoef(test_label_np, test_pred_binary)
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_threshold = threshold
            except Exception as e:
                print_log(f"Error calculating MCC at threshold {threshold}: {e}", logger=logger)

        # 使用最佳阈值计算最终指标
        test_pred_binary = (test_pred_prob_np > best_threshold).astype(int)
        try:
            mcc = matthews_corrcoef(test_label_np, test_pred_binary)
            precision = precision_score(test_label_np, test_pred_binary, zero_division=0)
            recall = recall_score(test_label_np, test_pred_binary, zero_division=0)
            f1 = f1_score(test_label_np, test_pred_binary, zero_division=0)
            if len(np.unique(test_label_np)) > 1:
                aucroc = roc_auc_score(test_label_np, test_pred_prob_np)
            else:
                aucroc = 0.0
                print_log(f"Warning: Only one class present in labels for epoch {epoch}. AUCROC set to 0.", logger=logger)

            print_log(f"Best Threshold: {best_threshold:.2f}, Val/Epoch_{epoch}/MCC: {mcc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUCROC: {aucroc:.4f}", logger=logger)

            # Add testing results to TensorBoard
            if val_writer is not None:
                val_writer.add_scalar('Val/Epoch/Best_Threshold', best_threshold, epoch)
                val_writer.add_scalar('Val/Epoch/MCC', mcc, epoch)
                val_writer.add_scalar('Val/Epoch/Precision', precision, epoch)
                val_writer.add_scalar('Val/Epoch/Recall', recall, epoch)
                val_writer.add_scalar('Val/Epoch/AUC-ROC', aucroc, epoch)
                val_writer.add_scalar('Val/Epoch/F1', f1, epoch)
        except Exception as e:
            print_log(f"Error calculating metrics with best threshold {best_threshold}: {e}", logger=logger)

        except ValueError as e:
            print_log(f"Error calculating metrics for epoch {epoch}: {e}", logger=logger)
            # Keep metrics as 0 or handle as needed

        final_metrics = Mcc_Metric(mcc) # Update final metric object

    # Ensure all processes wait until rank 0 is done
    if args.distributed:
        dist.barrier()

    # Return the metric object (all processes return it, but only rank 0's matters for saving)
    return final_metrics

