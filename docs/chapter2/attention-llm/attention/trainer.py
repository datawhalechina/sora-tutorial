from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def metric(y_test, y_pred):
    score = roc_auc_score(y_test, y_pred)
    return score


def train(model, optimizer, train_iter, valid_iter, config):
    model.train()
    total_batch = 0
    val_best_loss = float("inf")
    for epoch in range(1, config.num_epochs + 1):
        print(f"Epoch [{epoch}/{config.num_epochs}]")
        for i, (inputs, labels) in enumerate(train_iter, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            attn, outputs = model(inputs)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_batch += 1
            
        y_true = labels.data.cpu()
        y_pred = torch.max(outputs.data, 1)[1].cpu()
        train_acc = metric(y_true, y_pred)
        val_acc, val_loss = evaluate(model, valid_iter)
        
        if val_loss < val_best_loss:
            val_best_loss = val_loss
            torch.save(model.state_dict(), "best.pt")
        msg = f"Iter: {total_batch},  Train Loss: {loss.item():.2f},  Train Acc: {train_acc:.2f},  "
        msg += f"Val Loss: {val_loss:.2f},  Val Acc: {val_acc:.2f}"
        print(msg)
        model.train()


def evaluate(model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            texts = texts.to(device)
            labels = labels.to(device)
            attn, outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metric(labels_all, predict_all)
    return acc, loss_total / len(data_iter)


def test(model, test_iter):
    model.load_state_dict(torch.load("best.pt"))
    model.eval()
    test_acc, test_loss = evaluate(model, test_iter)
    return test_acc