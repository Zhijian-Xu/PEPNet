import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score
from tools import builder
from utils.logger import get_logger, print_log
from utils.config import cfg_from_yaml_file
from Bio import SeqIO, pairwise2

def parse_args():
    parser = argparse.ArgumentParser('Model Testing and Visualization')
    parser.add_argument('--subname', type=str, default='add_features', help='')
    parser.add_argument('--config', type=str, default='experiments/finetune_epitope/cfgs/add_features/config.yaml', help='config file')
    parser.add_argument('--ckpts', type=str, default='experiments/finetune_epitope/cfgs/add_features/ckpt-best.pth', help='path to checkpoint file')
    parser.add_argument('--log_dir', type=str, default='experiments/finetune_epitope/cfgs/add_features/log/', help='log directory')
    parser.add_argument('--threshold', type=float, default=None, help='User-specified threshold for classification')
    parser.add_argument('--subdir', type=str, default='finetune_epitope', help='log directory')
    
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use GPU')
    parser.add_argument('--use_alphafold3', action='store_true', help='use alphafold3 generated structures')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()
    args.config = f"experiments/{args.subdir}/cfgs/{args.subname}/config.yaml"
    args.ckpts = f"experiments/{args.subdir}/cfgs/{args.subname}/ckpt-best.pth"
    args.log_dir = f"experiments/{args.subdir}/cfgs/{args.subname}/log/"
    if args.use_gpu and args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        
    return args

def load_config(args):
    config = cfg_from_yaml_file(args.config)
    return config

def setup_logging(args):
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.log_dir, f'test_log_{timestamp}.txt')
    logger = get_logger(log_file)
    tb_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    
    return logger, writer

def build_dataloader(args, config):
    test_config = config.dataset.test.copy()
    _, test_dataloader = builder.dataset_builder_test(args, test_config)
    
    return test_dataloader

def test_model(model, dataloader, args, config, logger=None):
    """Test model and collect predictions and labels for each sample"""
    model.eval()
    print_log("Starting model testing...", logger=logger)
    
    all_pred_probs = []
    all_labels = []
    sample_metrics = []
    sample_ids = []
    if args.use_alphafold3: 
        print("**********use alphafold3 generated structures**********")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Testing", unit="batch")):
            # Move batch data to GPU if available
            device = torch.device("cuda" if args.use_gpu else "cpu")
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
            
            # Get predictions
            valid_probs, _, valid_epitopes, _, _, _ = model(batch)
            
            # Store results
            if valid_probs is not None and valid_epitopes is not None:
                valid_probs_list = valid_probs.cpu()
                valid_epitopes_list = valid_epitopes.cpu()
                # For per-sample metrics, extract sample identifiers if available
                if isinstance(batch, dict) and 'sample_id' in batch:
                    sample_id = batch['sample_id']
                else:
                    # If sample_id not in batch, use generic IDs
                    sample_ids = [f"sample_{idx}_{i}" for i in range(len(valid_epitopes))]
                
                if args.use_alphafold3: 
                    
                    try:
                        # Helper function to read FASTA files
                        def read_fasta(filepath):
                            """From FASTA file read sequence"""
                            try:
                                for record in SeqIO.parse(filepath, "fasta"):
                                    return str(record.seq)
                            except FileNotFoundError:
                                print(f"Warning: File not found {filepath}")
                                return None
                            return None
                        # 1. Load experimental epitope labels (ground truth)
                        labels_path = f"data/AsEP/AsEP_xyz_labels_pdbfixer_v5/{sample_id[0]}.npz"
                        if os.path.exists(labels_path):
                            exp_data = np.load(labels_path)
                            exp_epitopes = exp_data["epitope"]  # Ground truth from experimental structure
                        else:
                            print(f"Warning: Experimental epitope labels not found for {sample_id[0]}")
                            exp_epitopes = valid_epitopes_list.numpy()
                        
                        # 2. Read SEQRES sequence (AlphaFold3 sequence)
                        seqres_path = f"AsEP-dataset/data/AsEP/SEQRES_fasta/{sample_id[0]}_A_seqres.fasta"
                        seqres_seq = read_fasta(seqres_path)
                        
                        # 3. Read ATMSEQ sequence (experimental sequence)  
                        atmseq_path = f"AsEP-dataset/ATMSEQ_fasta/{sample_id[0]}_A.fasta"
                        atmseq_seq = read_fasta(atmseq_path)
                        
                        # 4. Perform alignment and mapping
                        if seqres_seq and atmseq_seq and len(valid_probs_list) > 0:
                            # Perform global alignment between SEQRES and ATMSEQ
                            alignments = pairwise2.align.globalxx(seqres_seq, atmseq_seq)
                            if alignments:
                                top_aln = alignments[0]
                                aligned_s1, aligned_s2 = top_aln[0], top_aln[1]
                                
                                # Map predicted scores from SEQRES positions to ATMSEQ positions
                                seqres_probs = valid_probs_list.numpy()
                                aligned_probs = []
                                s1_idx = 0  # SEQRES sequence and scores index
                                s2_idx = 0  # ATMSEQ sequence index
                                
                                for i in range(len(aligned_s1)):
                                    char_s1 = aligned_s1[i]
                                    char_s2 = aligned_s2[i]
                                    
                                    if char_s2 != '-':  # ATMSEQ position is not a gap
                                        if char_s1 != '-':  # SEQRES position is not a gap
                                            if s1_idx < len(seqres_probs):
                                                aligned_probs.append(seqres_probs[s1_idx])
                                            else:
                                                aligned_probs.append(0.0)  # Default value if index out of range
                                            s1_idx += 1
                                        else:
                                            # SEQRES has gap, ATMSEQ has residue - assign default score
                                            aligned_probs.append(0.0)
                                        s2_idx += 1
                                    else:  # ATMSEQ position is a gap
                                        if char_s1 != '-':
                                            s1_idx += 1
                                
                                # Update predictions and labels
                                if len(aligned_probs) == len(exp_epitopes):
                                    valid_probs_list = torch.tensor(aligned_probs)
                                    valid_epitopes_list = torch.tensor(exp_epitopes)
                                    # print(f"Successfully aligned {sample_id}: SEQRES({len(seqres_seq)}) -> ATMSEQ({len(atmseq_seq)}) -> scores({len(aligned_probs)})")
                                else:
                                    print(f"Warning: Length mismatch for {sample_id[0]}. ATMSEQ length: {len(atmseq_seq)}, aligned scores: {len(aligned_probs)}, exp epitopes: {len(exp_epitopes)}")
                                    valid_epitopes_list = torch.tensor(exp_epitopes)
                            else:
                                print(f"Warning: Could not align sequences for {sample_id[0]}")
                                valid_epitopes_list = torch.tensor(exp_epitopes)
                        else:
                            print(f"Warning: Missing sequence data for {sample_id[0]}")
                            if 'exp_epitopes' in locals():
                                valid_epitopes_list = torch.tensor(exp_epitopes)
                    
                    except ImportError:
                        print("Warning: BioPython not available, skipping AlphaFold3 alignment")
                    except Exception as e:
                        print(f"Error during AlphaFold3 alignment for {sample_id[0]}: {e}")

                # For overall dataset metrics
                all_pred_probs.append(valid_probs_list)
                all_labels.append(valid_epitopes_list)
                
                # Calculate metrics for this sample
                sample_pred = valid_probs_list.numpy()
                sample_label = valid_epitopes_list.numpy()
                
                # Store for later analysis
                sample_ids.append(sample_id)
                sample_metrics.append({
                    'pred': sample_pred,
                    'label': sample_label,
                    'id': sample_id
                })
    
    # Concatenate all results
    if all_pred_probs:
        test_pred = torch.cat(all_pred_probs, dim=0)
        test_label = torch.cat(all_labels, dim=0)
    else:
        test_pred = torch.empty((0,))
        test_label = torch.empty((0,))
    
    return test_pred, test_label, sample_metrics

def calculate_metrics(test_pred, test_label, logger=None, user_threshold=None):
    """Calculate metrics for entire dataset at various thresholds"""
    print_log("Calculating metrics across thresholds...", logger=logger)
    
    test_pred_np = test_pred.numpy()
    test_label_np = test_label.numpy()
    
    # Calculate metrics across different thresholds
    thresholds = np.arange(0.0, 1.01, 0.01)
    metrics_by_threshold = []
    
    for threshold in thresholds:
        test_pred_binary = (test_pred_np > threshold).astype(int)
        try:
            mcc = matthews_corrcoef(test_label_np, test_pred_binary)
            precision = precision_score(test_label_np, test_pred_binary, zero_division=0)
            recall = recall_score(test_label_np, test_pred_binary, zero_division=0)
            f1 = f1_score(test_label_np, test_pred_binary, zero_division=0)
            acc = accuracy_score(test_label_np, test_pred_binary)
            if len(np.unique(test_label_np)) > 1:
                aucroc = roc_auc_score(test_label_np, test_pred_np)
            else:
                aucroc = 0.0
            
            metrics_by_threshold.append({
                'threshold': threshold,
                'mcc': mcc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'acc': acc,
                'aucroc': aucroc
            })
        except Exception as e:
            print_log(f"Error calculating metrics at threshold {threshold}: {e}", logger=logger)
    
    best_metrics = max(metrics_by_threshold, key=lambda x: x['mcc'])
    print_log(f"Best metrics at threshold {best_metrics['threshold']:.2f}:", logger=logger)
    print_log(f"MCC: {best_metrics['mcc']:.4f}", logger=logger)
    print_log(f"Precision: {best_metrics['precision']:.4f}", logger=logger)
    print_log(f"Recall: {best_metrics['recall']:.4f}", logger=logger)
    print_log(f"F1: {best_metrics['f1']:.4f}", logger=logger)
    print_log(f"Accuracy: {best_metrics['acc']:.4f}", logger=logger)
    print_log(f"AUC-ROC: {best_metrics['aucroc']:.4f}", logger=logger)

    if user_threshold:
        closest = min(metrics_by_threshold, key=lambda x: abs(x['threshold'] - user_threshold))
        used_metrics = closest
        print_log(f"Using user-specified threshold {used_metrics['threshold']:.2f}:", logger=logger)
        print_log(f"MCC: {used_metrics['mcc']:.4f}", logger=logger)
        print_log(f"Precision: {used_metrics['precision']:.4f}", logger=logger)
        print_log(f"Recall: {used_metrics['recall']:.4f}", logger=logger)
        print_log(f"F1: {used_metrics['f1']:.4f}", logger=logger)
        print_log(f"Accuracy: {used_metrics['acc']:.4f}", logger=logger)
        print_log(f"AUC-ROC: {used_metrics['aucroc']:.4f}", logger=logger)
    else:
        used_metrics=None


    return metrics_by_threshold, best_metrics, used_metrics

def calculate_per_sample_metrics(sample_metrics, best_threshold, logger=None):
    """Calculate metrics for each individual sample using the best threshold"""
    print_log(f"Calculating per-sample metrics using threshold {best_threshold:.2f}...", logger=logger)
    
    per_sample_results = []
    
    for sample in sample_metrics:
        sample_pred = sample['pred']
        sample_label = sample['label']
        sample_id = sample['id']
        
        # Apply threshold
        sample_pred_binary = (sample_pred > best_threshold).astype(int)
        
        # Calculate metrics if possible
        try:
            if len(np.unique(sample_label)) > 1 and len(np.unique(sample_pred_binary)) > 1:
                mcc = matthews_corrcoef(sample_label, sample_pred_binary)
            else:
                # If all predictions or all labels are the same, MCC is undefined
                mcc = 0.0
                
            precision = precision_score(sample_label, sample_pred_binary, zero_division=0)
            recall = recall_score(sample_label, sample_pred_binary, zero_division=0)
            f1 = f1_score(sample_label, sample_pred_binary, zero_division=0)
            acc = accuracy_score(sample_label, sample_pred_binary)
            
            if len(np.unique(sample_label)) > 1:
                aucroc = roc_auc_score(sample_label, sample_pred)
            else:
                aucroc = 0.0
                
            per_sample_results.append({
                'id': sample_id,
                'mcc': mcc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'acc': acc,
                'aucroc': aucroc,
                'pred': sample_pred,
                'label': sample_label
            })
        except Exception as e:
            print_log(f"Error calculating metrics for sample {sample_id}: {e}", logger=logger)
            per_sample_results.append({
                'id': sample_id,
                'mcc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'acc': 0.0,
                'aucroc': 0.0,
                'pred': sample_pred,
                'label': sample_label
            })
    
    # Sort by MCC for easier analysis
    per_sample_results.sort(key=lambda x: x['mcc'], reverse=True)
    
    return per_sample_results

def visualize_results(metrics_by_threshold, best_metrics, per_sample_results, args, logger=None):
    """Generate visualizations of testing results"""
    print_log("Generating visualizations...", logger=logger)
    
    # Create visualization directory
    vis_dir = os.path.join(args.log_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. MCC vs Threshold curve
    thresholds = [m['threshold'] for m in metrics_by_threshold]
    mcc_values = [m['mcc'] for m in metrics_by_threshold]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mcc_values, marker='o', markersize=3)
    plt.axvline(x=best_metrics['threshold'], color='r', linestyle='--', 
                label=f'Best Threshold: {best_metrics["threshold"]:.2f}')
    plt.grid(True)
    plt.xlabel('Threshold')
    plt.ylabel('MCC')
    plt.title('MCC vs Threshold')
    plt.legend()
    plt.tight_layout()
    
    mcc_threshold_path = os.path.join(vis_dir, 'mcc_vs_threshold.png')
    plt.savefig(mcc_threshold_path)
    print_log(f"Saved MCC vs Threshold plot to {mcc_threshold_path}", logger=logger)
    plt.close()
    
    # 2. Per-sample MCC histogram
    per_sample_mcc = [result['mcc'] for result in per_sample_results]
    
    plt.figure(figsize=(10, 6))
    plt.hist(per_sample_mcc, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(per_sample_mcc), color='r', linestyle='--', 
                label=f'Mean MCC: {np.mean(per_sample_mcc):.4f}')
    plt.grid(True, alpha=0.3)
    plt.xlabel('MCC')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of MCC Across Samples')
    plt.legend()
    plt.tight_layout()
    
    mcc_hist_path = os.path.join(vis_dir, 'sample_mcc_histogram.png')
    plt.savefig(mcc_hist_path)
    print_log(f"Saved sample MCC histogram to {mcc_hist_path}", logger=logger)
    plt.close()
    
    # 3. Top 5 good cases and bad cases
    good_cases = per_sample_results[:5]  # Top 5 by MCC
    bad_cases = per_sample_results[-5:]  # Bottom 5 by MCC
    
    print_log("\nTop 5 Good Cases:", logger=logger)
    for i, case in enumerate(good_cases):
        print_log(f"{i+1}. Sample {case['id']}: MCC={case['mcc']:.4f}, F1={case['f1']:.4f}, AUC={case['aucroc']:.4f}", logger=logger)
        # Save predictions and ground truth for good cases
        save_path = os.path.join(args.log_dir, f"{case['id']}_good.npz")
        np.savez(save_path, pred=case['pred'], label=case['label'])
        print_log(f"Saved good case {case['id']} to {save_path}", logger=logger)
    
    print_log("\nTop 5 Bad Cases:", logger=logger)
    for i, case in enumerate(reversed(bad_cases)):
        print_log(f"{i+1}. Sample {case['id']}: MCC={case['mcc']:.4f}, F1={case['f1']:.4f}, AUC={case['aucroc']:.4f}", logger=logger)
        # Save predictions and ground truth for bad cases
        save_path = os.path.join(args.log_dir, f"{case['id']}_bad.npz")
        np.savez(save_path, pred=case['pred'], label=case['label'])
        print_log(f"Saved bad case {case['id']} to {save_path}", logger=logger)
    
    return vis_dir

def log_to_tensorboard(writer, metrics_by_threshold, best_metrics, per_sample_results, vis_dir):
    """Save results to tensorboard format"""
    # Log overall metrics
    writer.add_scalar('Test/Best_MCC', best_metrics['mcc'])
    writer.add_scalar('Test/Best_Threshold', best_metrics['threshold'])
    writer.add_scalar('Test/Best_F1', best_metrics['f1'])
    writer.add_scalar('Test/Best_Precision', best_metrics['precision'])
    writer.add_scalar('Test/Best_Recall', best_metrics['recall'])
    writer.add_scalar('Test/Best_AUC', best_metrics['aucroc'])
    writer.add_scalar('Test/Best_Accuracy', best_metrics['acc'])
    
    # Log MCC vs Threshold curve
    mcc_values = [m['mcc'] for m in metrics_by_threshold]
    for idx, threshold in enumerate([m['threshold'] for m in metrics_by_threshold]):
        writer.add_scalar('Test/MCC_by_Threshold', mcc_values[idx], global_step=int(threshold*100))
    
    # Log per-sample metrics
    for idx, result in enumerate(per_sample_results):
        writer.add_scalar(f'Samples/MCC', result['mcc'], global_step=idx)
        writer.add_scalar(f'Samples/F1', result['f1'], global_step=idx)
        writer.add_scalar(f'Samples/AUC', result['aucroc'], global_step=idx)
    
    # Add visualizations
    mcc_threshold_img = os.path.join(vis_dir, 'mcc_vs_threshold.png')
    mcc_hist_img = os.path.join(vis_dir, 'sample_mcc_histogram.png')
    
    if os.path.exists(mcc_threshold_img):
        img = plt.imread(mcc_threshold_img)
        writer.add_image('Visualizations/MCC_vs_Threshold', img.transpose(2, 0, 1))
    
    if os.path.exists(mcc_hist_img):
        img = plt.imread(mcc_hist_img)
        writer.add_image('Visualizations/Sample_MCC_Histogram', img.transpose(2, 0, 1))
    
    # Close writer
    writer.close()

def main():
    args = parse_args()
    config = load_config(args)
    logger, writer = setup_logging(args)

    print_log("Arguments:", logger=logger)
    for arg in vars(args):
        print_log(f"  {arg}: {getattr(args, arg)}", logger=logger)
    
    if not args.ckpts or not os.path.exists(args.ckpts):
        print_log(f"Error: Checkpoint file not found at {args.ckpts}", logger=logger)
        return
     
    model = builder.model_builder(config.model)
    if args.use_gpu:
        model.cuda()
    
    print_log(f"Loading model weights from {args.ckpts}", logger=logger)

    try:
        model.load_state_dict(torch.load(args.ckpts, map_location=torch.device("cuda" if args.use_gpu else "cpu"))['base_model'],strict=False) 
    except Exception as e:
        print_log(f"Error loading model: {e}", logger=logger)
        return


    test_dataloader = build_dataloader(args, config)
    
    test_pred, test_label, sample_metrics = test_model(model, test_dataloader, args, config, logger)

    metrics_by_threshold, best_metrics, _ = calculate_metrics(test_pred, test_label, logger, args.threshold)

    test_pred_flat = test_pred.detach().cpu().numpy()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.hist(test_pred_flat, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of test predictions')
    plt.xlabel('Prediction value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_pred_histogram.png", dpi=300)
    plt.close()

    mean = np.mean(test_pred_flat)
    std = np.std(test_pred_flat)
    min_val = np.min(test_pred_flat)
    max_val = np.max(test_pred_flat)
    percentiles = np.percentile(test_pred_flat, [25, 50, 75])

    print("📊 Prediction Statistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std Dev: {std:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    print(f"  25th Percentile: {percentiles[0]:.4f}")
    print(f"  Median (50th): {percentiles[1]:.4f}")
    print(f"  75th Percentile: {percentiles[2]:.4f}")

    # Calculate per-sample metrics
    per_sample_results = calculate_per_sample_metrics(sample_metrics, args.threshold, logger)
    
    # Visualize results
    vis_dir = visualize_results(metrics_by_threshold, best_metrics, per_sample_results, args, logger)

    # Log to tensorboard with data type in path
    log_to_tensorboard(writer, metrics_by_threshold, best_metrics, per_sample_results, vis_dir)
    
    print_log("\nTesting and visualization complete.", logger=logger)

if __name__ == "__main__":
    main()