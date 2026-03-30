from pathlib import Path# for finding path and files names
import torch # pytorch  for creating tensors, build neural networks -- the main deep library
import torch.nn as nn # neural network
from torch.utils.data import DataLoader, random_split # it helps us loader all samples to the model at once, we give a few at a time

from model import BetterWindCNN #brings the convulution neural network
from dataset import load_wind_time_series, WindForecastDataset


#dataset = the textbook with examples
#model = the student brain
#train=the student studies
#loss= how wrong the student is
#optimiser= the correction mathod
#epochs=how many times the student reviews the textbook

INPUT_STEPS =2 # two past time steps as input
TARGET_OFFSET =1 # predict the 1 step in the future
TRAIN_RATIO =0.8 # training rate
BATCH_SIZE=4 #the model sees 4 samples at a time before updating itself
LEARNING_RATE=0.001# the correction steps are (big value leads to faster but unstable solutions)
NUM_EPOCHS=20 # the model will go through the whole trainning dataset 20 times
MODEL_FILENAME= "wind_forecast_cnn.pth"


# Choose devise
if torch.cuda.is_available():
	device =torch.device("cuda")
	print("Using nvidia gpu")
else:
	device =torch.device("cpu")
	print("Using CPU")
	
	
# Create Paths
project_root=Path(__file__).reslove().parent.parent# this mean go the previous of the previous path 
processed_dir=project/ "data" / "processed"
model_dir=project_root/ "saved_models"
model_dir.mkdir(parents=True, exist_ok=True)

print("Processed data folder:", processed_dir)
print("Model save folder",model_dir)



# Load data
data, __, latitudes, longitudes = load_wind_time_series(processed_dir)# __, ==> i am receiving this value but i am not planning to use them

dataset=WindForecast(data=data, input_steps=INPUT_STEPS, target_offset=TARGET_OFFSET)

if len(dataset)==0:
	raise ValueError("Dataset is empty.")
	
print("Number of dataset:", len(dataset))	# show how many training examples were created 


#Split data
train_size = int(TRAIN_RATIO* len(dataset)) # how many samples go to the training
val_size =len(dataset) - train_size # the rest of validation

if train_size==0:
	train_size=1
	val_size =len(dataset)-1 # of training accidentaly becomes zero , fix it

if val_size ==0 :
	val_size =1
	train_size =len(dataset) -1 # if validation soze becomes zero fir it
	
train_dataset, val_dataset =random_split(dataset, [train_size, val_size]) # Random split the data to training part, validate part 

train_loader=DataLoader(train_dataset, batch_size=min(train_dataset), shuffle =True)		# create a loader for training data
val_loader=DataLoader(val_dataset, batch_size=min(BATCH_SIZE),len(val_dataset), shuffle =False)	# create a loader for validating data

print("Training samples", len(train_dataset))
print("Validation samples", len(val_dataset))



# Build model
sample_x, sample_y =dataset[0] # take the first simple from the dataset

in_channels = sample_x.shape[0] # input
out_channels =sample_y.shape[0] # target

print("Input channels: ", in_channels)
print("Output channels: ", out_channels)


model=BetterWindCNN(in_channels=in_channels,out_channels=out_channels).to(device)# create the convulution neural network


#Loss and optimizer
criterion= nn.MSELoss() # choose the loss function -- loss function tell us how wrong the model is
                        #MSE (Mean Square Error) is common for regression problems
optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # optimiser - to update weights and correct the problem
	
	
# 	Train loop
for epoch in range(NUM_EPOCHS): # repeat training for many epochs
	model.train() # put the model in training mode
	train_loss=0.0 # stay counting total training loss for this epoch
	
	for x_batch, y_batch in train_loader:
		x_batch =x_batcg.to(device) # move the batch to the same davice as model
		y_batch=y_batch.to(device)
		
		optimizer.zero_grad() # clear old gradiens calculating new ones 
		
		predictions=model(x_batch) #send the input through the model to get predictions
		
		loss=criterion(predictions,y_batch) # compare model predictions with true answer
		                                    # smaller loss = better predictions 
		loss.backward()                     # compute gradients, it is called backpropagation -- it tells the model how to predict the error
		
		optimizer.step()                   # learning step - update model parameter using gradients
		
		train_loss+=loss.item()            #add the batch's loss to the total epoch loss 
		
	train_loss=train_loss /len(train_loader)# compute the averga training loss across all batches 
 
    model.eval()#===>put the model in evaluation mode -- "now i am not training , only testing"
    val_loss =0.0
    
    
    with torch.no.grad():# validation srage- no update gratients now
		for x_batch, y_batcj in val_loader:
			x_batch =x_batch.to(device)
			y_batch=y_batch.to(device)
			
			predictions=model(x_batch)# make prediction
			loss=criterion(prediction,y_batch)# make validation error
			
			val_l0ss+=loss.item()# add validation batch loss to total validation loss 
			
	val_loss=val_loss /len(val_loader)  # compute the avergae validation loss
	
	print(f"Epoch {epoch+1}/{NUM_EPOCHS} -Train Loss{}" -Val Loss:{val_loss.6f})	# showning training progress 	    

# Save model
model_path =model_dir /MODEL_FILENAME
torch.save(model.state_dict(), model_path)

print("Training finishes. Saved in ", model_path)


	
