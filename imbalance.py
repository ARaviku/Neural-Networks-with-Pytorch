
from dataset import DogCatDataset
from train import evaluate_loop, train
from transfer import load_pretrained

def per_class_accuracy(y_true, y_pred, num_classes=2):
    m_size = y_pred.size()[0]
    correct_class0 = 0
    correct_class1 = 0
    class0 = 0
    class1 = 0
    for i in range(m_size):
      if y_true[i] == 0:
        class0 += 1
        if y_pred[i] == y_true[i]:
            correct_class0 += 1
      else:
        class1 += 1
        if y_pred[i] == y_true[i]:
            correct_class1 += 1
    accuracy_0 = correct_class0/class0
    accuracy_1 = correct_class1/class1
    return [accuracy_0, accuracy_1]


def precision(y_true, y_pred):
    m_val = y_pred.size()[0]
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    m_val = y_pred.size()[0]
    for i in range(m_val):
      if y_true[i] == y_pred[i]:
        true_pos += 1
      else:
        if y_pred[i] == 1:
          false_pos += 1;
    precision_val = true_pos/(true_pos + false_pos)
    return precision_val
    # return 0.0


def recall(y_true, y_pred):
    m_val = y_pred.size()[0]
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    m_val = y_pred.size()[0]
    for i in range(m_val):
      if y_true[i] == y_pred[i]:
        true_pos += 1
      else:
        if y_true[i] == 1:
          false_neg += 1;
    recall_val = true_pos/(true_pos + false_neg)
    return recall_val
    # return 0.0


def f1_score(y_true, y_pred):
    precision_val = precision (y_true, y_pred)
    recall_val = recall(y_true,y_pred) 
    m_val = y_pred.size()[0]
    for i in range(m_val):
      f1_val = (2*precision*recall)/(precision + recall)
    return f1_val


def compute_metrics(dataset, model):
    y_true, y_pred, _ = evaluate_loop(dataset.val_loader, model)
    print('Per-class accuracy: ', per_class_accuracy(y_true, y_pred))
    print('Precision: ', precision(y_true, y_pred))
    print('Recall: ', recall(y_true, y_pred))
    print('F1-score: ', f1_score(y_true, y_pred))


if __name__ == '__main__':
    # model with normal cross-entropy loss
    config = {
        'dataset_path': 'data/images/dogs_vs_cats_imbalance',
        'batch_size': 4,
        # 'ckpt_force': True,
        'ckpt_path': 'checkpoints/imbalance',
        'plot_name': 'Imbalance',
        'num_epoch': 5,
        'learning_rate': 1e-3,
        'momentum': 0.9,
    }
    dataset = DogCatDataset(config['batch_size'], config['dataset_path'])
    model = load_pretrained(num_classes=2)
    train(config, dataset, model)
    compute_metrics(dataset, model)

    # model with weighted cross-entropy loss
    config = {
        'ckpt_path': 'checkpoints/imbalance_weighted',
        'plot_name': 'Imbalance-Weighted',
        'num_epoch': 5,
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'use_weighted': True,
    }
    model_weighted = load_pretrained(num_classes=2)
    train(config, dataset, model_weighted)
    compute_metrics(dataset, model_weighted)