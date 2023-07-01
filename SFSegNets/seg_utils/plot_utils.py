import matplotlib.pyplot as plt

def plot_iou(logs):
    plt.figure(figsize=(20,8))
    plt.plot(logs.index.tolist(), logs.iou_score.tolist(), lw=3, label = 'Train')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('IoU Score', fontsize=20)
    plt.title('IoU Score Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('iou_score_plot.png')
    plt.show()

def plot_dice(logs):
    plt.figure(figsize=(20,8))
    plt.plot(logs.index.tolist(), logs.dice_loss.tolist(), lw=3, label = 'Train')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Dice Loss', fontsize=20)
    plt.title('Dice Loss Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('dice_loss_plot.png')
    plt.show()