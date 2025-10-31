from trainer import Trainer
from utils.data import get_MNIST_data, get_VNFood21_data
from models.GoogleLeNet import GoogleLeNet
from models.LeNet import LeNet
from models.pretrained_resnet import PretrainedResnet as ResNet50
#from models.ResNet18 import RestNet18
import torch.optim as optim
import torch.nn as nn
import torch 
import argparse
def get_exercise(name):
    if name =="ex1":
        model = LeNet()
        train_loader,val_loader,test_loader = get_MNIST_data(batch_size=32,num_workers=2)
    elif name =="ex2":
        model = GoogleLeNet(num_classes=21)
        train_loader,val_loader,test_loader = get_VNFood21_data(batch_size=32,num_workers=2)
    elif name =="ex3":
        model = RestNet18(num_classes=21)
        train_loader,val_loader,test_loader = get_VNFood21_data(batch_size=32,num_workers=2)
    else:
        model = ResNet50()
        train_loader,val_loader,test_loader = get_VNFood21_data(batch_size=32,num_workers=2)
    return model, train_loader,val_loader,test_loader

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exercise", type=str, default="ex1", help="Choose exercise to run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model,train_loader,val_loader,test_loader = get_exercise(args.exercise)
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model,optimizer,criterion,device)
    EPOCHS = 5
    for epoch in range(EPOCHS):
        train_loss,train_metrics,val_loss,val_metrics = trainer.train_and_validate(train_loader,val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}")
        ## Print accuracy and f1-score
        print(f"  Train: loss={train_loss:.4f}, acc={train_metrics['accuracy']:.4f}, f1={train_metrics['f1_macro']:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_metrics['accuracy']:.4f}, f1={val_metrics['f1_macro']:.4f}")
        print("-"*60,'\n')
    
    if args.exercise =='ex1':
        torch.save(model.state_dict(), "LeNet.pth")
    elif args.exercise =='ex2':
        torch.save(model.state_dict(), "GoogleLeNet.pth")
    elif args.exercise =='ex3':
        torch.save(model.state_dict(), "ResNet18.pth")
    else:
        torch.save(model.state_dict(), "ResNet50.pth")
    print("✅ Training complete. Model saved!")

    test_loss, test_metrics, test_performance = trainer.test(test_loader)
    print(f'\n\nFinal state')
    print(f'Test: loss={test_loss:.4f}, acc={test_metrics["accuracy"]:.4f}, f1={test_metrics["f1_macro"]:.4f}')
    print('Confusion Matrix on test set:')
    print(test_performance)
if __name__ == "__main__":
    main()