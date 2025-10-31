
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def compute_metrics(self, y_true, y_pred):
        """accuracy, precision, recall, f1 macro"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)
        }
    def compute_confusion_matrix(self, y_true, y_pred):
        '''confusion matrix and print'''
        cm = confusion_matrix(y_true, y_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
        # disp.plot(cmap=plt.cm.Blues)
        # plt.title("Confusion Matrix (MNIST)")
        # plt.xlabel("Predicted label")
        # plt.ylabel("True label")
        # plt.show()
        return cm
    def train_and_validate(self, train_loader, val_loader):
        """train + val"""
        self.model.train()
        total_loss = 0
        y_true_train, y_pred_train = [], []

        ################## TRAINING ##################
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())


        train_loss = total_loss / len(train_loader.dataset)
        train_metrics = self.compute_metrics(y_true_train, y_pred_train)

        ################## VALIDATING ##################
        self.model.eval()
        val_loss = 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_metrics = self.compute_metrics(y_true_val, y_pred_val)

        return train_loss, train_metrics, val_loss, val_metrics

    def test(self,test_loader):
        self.model.eval()
        test_loss = 0
        y_true_test, y_pred_test = [], []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(preds.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        test_metrics = self.compute_metrics(y_true_test, y_pred_test)
        test_performance = self.compute_confusion_matrix(y_true_test,y_pred_test)

        return test_loss, test_metrics, test_performance
