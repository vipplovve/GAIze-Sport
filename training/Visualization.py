import os
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use('Agg')

def plot_loss_curves(train_losses, val_losses=None, title="Loss Curve", filename="loss.png", save_dir="checkpoints"):
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    
    if val_losses:
        plt.plot(val_losses, label='Val Loss', color='orange', linewidth=2)
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def plot_psnr_curve(psnr_values, title="PSNR Curve", filename="psnr.png", save_dir="checkpoints"):
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(psnr_values, label='PSNR (dB)', color='green', linewidth=2)
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def plot_accuracy_curves(train_accs, val_accs, title="Accuracy Curve", filename="accuracy.png", save_dir="checkpoints"):
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_accs, label='Train Acc (%)', color='green', linewidth=2)
    plt.plot(val_accs, label='Val Acc (%)', color='red', linewidth=2)
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", filename="confusion_matrix.png", save_dir="checkpoints"):
    try:
        from sklearn.metrics import confusion_matrix
        import numpy as np
    except ImportError:
        print("[Visualization] sklearn not found. Please install scikit-learn for Confusion Matrix.")
        return None

    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    
    fmt_norm = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_text = f"{cm[i, j]}\n({format(cm_norm[i, j], fmt_norm)})"
            ax.text(j, i, cell_text,
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def plot_gan_loss(g_losses, d_losses, title="GAN Loss", filename="gan_loss.png", save_dir="checkpoints"):
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label='Generator Loss', color='purple', alpha=0.8)
    plt.plot(d_losses, label='Discriminator Loss', color='green', alpha=0.8)
        
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)
