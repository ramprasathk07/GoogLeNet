import torch 
import torch.nn as nn
from tqdm import tqdm
from models.GoogLeNet_model import GoogLeNet
from dataset import *
from torchvision import transforms
import yaml 

with open('/_.yaml') as f:
    config = yaml.safe_load(f)

num_epoch = 50 
img_size = 224

df = pd.read_csv('./')

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print('Using Device',device)

model = GoogLeNet(num_class=5).to(device=device)

train_dataloader= get_data(images_class=df['labels'].tolist(),
                                           images_path=df['path'].tolist(),
                                           transform=data_transform['train'])

val_dataloader= get_data(images_class=df['labels'].tolist(),
                                           images_path=df['path'].tolist(),
                                           transform=data_transform['val'])

train_loader = torch.utils.data.DataLoader(train_dataloader,
                                            batch_size=config['train_bs'],
                                            shuffle=True,
                                            pin_memory=True,
                                            collate_fn=train_dataloader.collate_fn)

val_loader = torch.utils.data.DataLoader(val_dataloader,
                                            batch_size=config['val_bs'],
                                            shuffle=False,
                                             pin_memory=True,
                                             collate_fn=val_dataloader.collate_fn)

optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps = 1e-9)
loss_fn = nn.CrossEntropyLoss() 

initial_epoch = 0

train_accuracies,train_losses = [],[]
val_accuracies,val_losses = [],[]

for epoch in range(num_epoch):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
   
    with tqdm(total=len(train_loader),desc = f"Training Epoch:{epoch+1}") as progress_bar:
        for step,(image,labels) in enumerate(train_loader):
            train_step,val_step = 0,0 
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            preds = model(image.to(device)) 
            lables = lables.type(torch.float32).to(device)
            loss = loss_fn(preds, labels)

            train_losses.append(loss.cpu().data.numpy())

            loss.backward()
            optimizer.step()
            
            _,predicted_classifier = torch.max(preds, 1)
            train_step += labels.size(0)

            train_correct += (predicted_classifier == labels.to(predicted_classifier.device).argmax(dim=1)).sum().item()
            tacc = train_correct / train_step

            progress_bar.set_postfix({"Loss": f"{loss:.4f}","Accuracy": f"{tacc:.4f}"})
            progress_bar.update()
    
    train_loss /=len(train_loader)
    train_acc = 100 * train_correct / train_step
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    val_loss, val_correct, val_total = 0.0, 0, 0

    model.eval()
    with tqdm(total=len(val_loader), desc=f"Validation Phase: Epoch {epoch+1}") as progress_bar:
        for image, lables in val_loader:
            image = image.to(device)
            lables = lables.type(torch.float32).to(device)

            with torch.no_grad():
                logits_v = model(image)

            loss =loss_fn(logits_v,image)
            val_losses.append(loss.cpu().data.numpy())

            val_loss+=loss.item()
            val_step += labels.size(0)

            _, predicted_classifier = torch.max(logits_v, 1)
            val_correct += (predicted_classifier == labels.to(predicted_classifier.device).argmax(dim=1)).sum().item()
            vall = val_correct / val_step
            progress_bar.set_postfix({"Loss": f"{loss:.4f}","Accuracy": f"{vall:.4f}"})
            progress_bar.update()