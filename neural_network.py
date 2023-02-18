# %%
import yaml
import torch 
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter 
from modelling import *

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df=pd.read_csv('clean_tabular_data.csv')
        self.df = self.df.drop(columns=['ID','Category','Title','Description','Amenities','Location','url','Unnamed: 19'], axis = 1)
        # Select the rows that have 'Somerford Keynes England United Kingdom'
        self.df = self.df[self.df['guests'] != 'Somerford Keynes England United Kingdom']
        self.df = self.df.apply(pd.to_numeric)
        self.feature = self.df.drop(columns=['Price_Night'], axis = 1)
        

        self.labels = self.df['Price_Night']
        
    # Not dependent on index
    def __getitem__(self, index):
        feature = self.feature.iloc[index]
        labels = self.labels.iloc[index]


       
        feature=torch.tensor(feature,dtype=torch.float32).unsqueeze(0)
        label=torch.tensor(labels, dtype=torch.float32)

        feature=torch.nn.functional.normalize(feature)
        

    
        #Normalise data
        # mean, std, var = torch.mean(feature), torch.std(feature), torch.var(feature)
        # feature  = (feature-mean)/std

        return (feature, label)

    def __len__(self):
        return len(self.feature)


def train(model,config,train_loader,epochs = 20 ):

    
    writer=SummaryWriter()

    optimiser=eval(config['optimiser'])

    opt = optimiser(model.parameters(), lr = config['learning_rate'] ) #then you increase the learning rate to 0.01
    loss_fn = torch.nn.MSELoss()
    print(train_loader)

    batch_index=0
    for epoch in range(epochs):
        for batch in train_loader:
            features, labels = batch
            prediction=model(features)
            loss = loss_fn(prediction, labels) 
            #acc = model.evaluate(features, labels)[1]

            # Apply backprop 

            loss.backward() # do back proprogation
            print(loss.item())
            opt.step()  # this will update weight and biases
            opt.zero_grad() # resets the gradients perform gradient dewcent

            writer.add_scalar("loss", loss.item(), batch_index) #track data for tensor board 
            batch_index += 1

def get_nn_config():
    with open('nn_config.yaml') as f:
        config = yaml.safe_load(f)
    return config
    

class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.layer = torch.nn.Sequential(torch.nn.Linear(11, self.config['hidden_layer_width']))

        for i in range(self.config['model_depth']):
            self.layer.add_module(f'fc{i}', torch.nn.Linear(in_features=self.config['hidden_layer_width'], out_features=self.config['hidden_layer_width']))
            self.layer.add_module(f'relu{i}', torch.nn.ReLU())
        self.layer.add_module(f'fc{self.config["model_depth"]}', torch.nn.Linear(in_features=self.config['hidden_layer_width'], out_features=1))


    def forward(self, x):
       x = self.layer(x)      
       return x



if __name__ == "__main__":
    

    dataset=AirbnbNightlyPriceImageDataset()
    # features , labels = dataset[1]
    # print(features)
   


    batch_size=8
    test_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size)

    validation_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - validation_size
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
    validation_loader = DataLoader(validation_dataset,batch_size=batch_size)

    config= get_nn_config()

    #print((config['optimiser']))



    model = Net(config=config)


    train(model,config, train_loader)


# %%
