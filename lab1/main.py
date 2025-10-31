from trainer import Trainer
from utils.data import get_data
from models.mlp import MLP1, MLP2
import torch.optim as optim
import torch.nn as nn
import torch 
import argparse
def get_model(name):
    if name =="mlp1":
        model = MLP1(input_size=28*28,output_size=512)
    else:
        model = MLP2(input_size=28*28,hidden1=512,hidden2=256,output_size=10)
    return model

def main():
    parser = argparse.ArgumentParser(description="Train model on MNIST")
    parser.add_argument("--model", type=str, default="mlp1", help="MLP for exercise 1 or 2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader = get_data()
    model = get_model(args.model).to(device)
    optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
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
    
    if args.model =='mlp1':
        torch.save(model.state_dict(), "mlp1_mnist.pth")
    else:
        torch.save(model.state_dict(), "mlp2_mnist.pth")
    print("✅ Training complete. Model saved!")

    test_loss, test_metrics, test_performance = trainer.test(test_loader)
    print(f'\n\nFinal state')
    print(f'Test: loss={test_loss:.4f}, acc={test_metrics["accuracy"]:.4f}, f1={test_metrics["f1_macro"]:.4f}')
    print('Confusion Matrix on test set:')
    print(test_performance)
if __name__ == "__main__":
    main()