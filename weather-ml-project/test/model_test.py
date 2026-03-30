import torch
import torch as nn




class ResudualBlock(nn.Module) # this create a smali building block on neural network
""" This create a small building block
"""
	def__init__(self,channels: int): # channels of feauture maps(like layers)
		super().__inti__()
		
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding =1) # this is a cpnvulution layer
		
		sel.relu =nn.ReLu() # thish is an activation function that removes negatives values, keeeps only usufull signs

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding =1) # this is a cpnvulution layer
       
	def forwnard( self ,x :torch.Tensor) -> torch.Tensor:
		identity =x
		
		out =self.conv1(x)
		out=self.relu(out)
		out=self.con2(out)
		
		out =out+identity
		
		out=self.relu(out)
		
		
		return out 



class BetterWindCNN(nn.Module):
	""" this is the full model
	"""
	def__init__(self, in_chanells, out_chanels=2, hiden_chanels=64):
		# in_channels -> how many inputs , out_channels-> what you predict, hidden_channels-> internal size of networks
    self.input_layer == nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padiing=1)
            )
   # more channels more feature more learning capacity
   #Residual Block
   self.res_block1=ResidualBlock(hidden_channels)
   self.res_block2=ResidualBlock(hidden_channels)
   self.res_block3=ResidualBlock(hidden_channels)
   
   #Output Layer
   self.output_layer = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
   
   
   #Forward Pass        
   def forward(self, x:torch.Tensor) ->torch.Tensor:
	    
