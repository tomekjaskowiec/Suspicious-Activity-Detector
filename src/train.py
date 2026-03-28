import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Model
from data_u import load_and_scale_data

train_loader, val_data, test_data = load_and_scale_data()
X_val, y_val = val_data
X_test, y_test = test_data

model = Model()
criterion = nn.BCELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
losses, accuracies = [], []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    losses.append(epoch_loss / len(train_loader))

    # Validation llop
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        preds = (y_val_pred > 0.5).float()
        acc = (preds == y_val).float().mean()
        accuracies.append(acc.item())
        print(f"Epoch {epoch:>3} | Train loss: {losses[-1]:.4f} | Acc: {acc:.4f}")




#test loop
model.eval()
with torch.no_grad():
    raw_out = model(X_test)
    predictions = (raw_out.view(-1) > 0.5).float()
    correct = (predictions == y_test.view(-1)).sum().item()
    test_accuracy = correct / y_test.size(0)

print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")









torch.save(model.state_dict(), "model_weights.pth")
plt.plot(losses, label='Loss')
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.show()