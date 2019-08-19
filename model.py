
import keras.layers.core as core
import keras.layers.convolutional as conv
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
import keras.metrics
from keras.models import load_model
from LossCheckPoint import LossModelCheckpoint


def OnehotNetwork(train_oneofkeyX,trainY,val_oneofkeyX,valY,
				#pre_train_total_path = 'model/pretrain.h5',
				train_time=None,compilemodels=None):
    
	Oneofkey_input = Input(shape= (train_oneofkeyX.shape[1],train_oneofkeyX.shape[2]))  #49*21=1029
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)
	
	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l1(0),border_mode="same")(Oneofkey_input)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(101,7,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)
		
		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		
		x = Dense(256,init='glorot_normal',activation='relu')(x)
		x = Dropout(0.298224)(x)
		
		x = Dense(128,init='glorot_normal',activation="relu")(x)
		x = Dropout(0)(x)

		Oneofkey_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer= l2(0.001))(x)
		
		OnehotNetwork = Model(Oneofkey_input,Oneofkey_output)
		
		optimization='Nadam'
		OnehotNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
	else:
		OnehotNetwork = load_model("model/"+str(train_time-1)+'model/OnehotNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/OnehotNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/OnehotNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/OnehotNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"onehotloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')		
		onehotfitHistory = OnehotNetwork.fit(train_oneofkeyX,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_oneofkeyX,valY))
		OnehotNetwork.save("model/"+str(train_time)+'model/1OnehotNetwork.h5')

	return OnehotNetwork


def OtherNetwork(train_physicalXo,trainY,val_physicalXo,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):
    
	physical_O_input = Input(shape=(train_physicalXo.shape[1],train_physicalXo.shape[2]))  #49*28=1372
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(physical_O_input)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)
		
		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		physical_O_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)

		OtherNetwork = Model(physical_O_input,physical_O_output)
		
		optimization='Nadam'
		OtherNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
	else:
		OtherNetwork = load_model("model/"+str(train_time-1)+'model/OtherNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/OtherNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/OtherNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/OtherNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Onetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')		
		OfitHistory = OtherNetwork.fit(train_physicalXo,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXo,valY))                
		OtherNetwork.save("model/"+str(train_time)+'model/1OtherNetwork.h5')
        
	return OtherNetwork


def PhysicochemicalNetwork(train_physicalXp,trainY,val_physicalXp,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_P_input = Input(shape=(train_physicalXp.shape[1],train_physicalXp.shape[2]))  #49*37=1813
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(physical_P_input)
		x = Dropout(0.3)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		physical_P_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)

		PhysicochemicalNetwork = Model(physical_P_input,physical_P_output)
		
		optimization='Nadam'
		PhysicochemicalNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
	else:
		PhysicochemicalNetwork = load_model("model/"+str(train_time-1)+'model/PhysicochemicalNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/PhysicochemicalNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/PhysicochemicalNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/PhysicochemicalNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Pnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')		
		PfitHistory = PhysicochemicalNetwork.fit(train_physicalXp,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXp,valY))                
		PhysicochemicalNetwork.save("model/"+str(train_time)+'model/1PhysicochemicalNetwork.h5')
        
	return PhysicochemicalNetwork    


def HydrophobicityNetwork(train_physicalXh,trainY,val_physicalXh,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_H_input = Input(shape=(train_physicalXh.shape[1],train_physicalXh.shape[2]))  #49*149=7301
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(physical_H_input)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.3)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(51,7,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		physical_H_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)

		HydrophobicityNetwork = Model(physical_H_input,physical_H_output)
		
		optimization='Nadam'
		HydrophobicityNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
	else:
		HydrophobicityNetwork = load_model("model/"+str(train_time-1)+'model/HydrophobicityNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/HydrophobicityNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/HydrophobicityNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/HydrophobicityNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Hnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')		
		HfitHistory = HydrophobicityNetwork.fit(train_physicalXh,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXh,valY))                
		HydrophobicityNetwork.save("model/"+str(train_time)+'model/1HydrophobicityNetwork.h5')
    	
	return HydrophobicityNetwork    


def CompositionNetwork(train_physicalXc,trainY,val_physicalXc,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_C_input = Input(shape=(train_physicalXc.shape[1],train_physicalXc.shape[2]))  #49*24=1176
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(physical_C_input)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)
		
		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		physical_C_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)

		CompositionNetwork = Model(physical_C_input,physical_C_output)
		
		optimization='Nadam'
		CompositionNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
	else:
		CompositionNetwork = load_model("model/"+str(train_time-1)+'model/CompositionNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/CompositionNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/CompositionNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/CompositionNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Cnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')		
		CfitHistory = CompositionNetwork.fit(train_physicalXc,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXc,valY))                     	
		CompositionNetwork.save("model/"+str(train_time)+'model/1CompositionNetwork.h5')
        
	return CompositionNetwork


def BetapropensityNetwork(train_physicalXb,trainY,val_physicalXb,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_B_input = Input(shape=(train_physicalXb.shape[1],train_physicalXb.shape[2]))  #49*37=1813
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(physical_B_input)
		x = Dropout(0.3)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		physical_B_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)

		BetapropensityNetwork = Model(physical_B_input,physical_B_output)
		
		optimization='Nadam'
		BetapropensityNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
	else:
		BetapropensityNetwork = load_model("model/"+str(train_time-1)+'model/BetapropensityNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/BetapropensityNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/BetapropensityNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/BetapropensityNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Bnetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')		
		BfitHistory = BetapropensityNetwork.fit(train_physicalXb,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXb,valY))                
		BetapropensityNetwork.save("model/"+str(train_time)+'model/1BetapropensityNetwork.h5')
        
	return BetapropensityNetwork


def AlphaturnpropensityNetwork(train_physicalXa,trainY,val_physicalXa,valY,
			#pre_train_total_path = 'model/pretrain.h5',
			train_time=None,compilemodels=None):

	physical_A_input = Input(shape=(train_physicalXa.shape[1],train_physicalXa.shape[2]))  #49*118=5782
	early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10)

	if (train_time==0):
		x = conv.Convolution1D(201,2,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(physical_A_input)
		x = Dropout(0.4)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(151,3,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.3)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(101,5,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.2)(x)
		x = Activation('relu')(x)

		x = conv.Convolution1D(51,7,init='glorot_normal',W_regularizer= l2(0),border_mode="same")(x)
		x = Dropout(0.1)(x)
		x = Activation('relu')(x)

		x = core.Flatten()(x)
		x = BatchNormalization()(x)
		physical_A_output = Dense(2,init='glorot_normal',activation='softmax',W_regularizer=l2(0.001))(x)

		AlphaturnpropensityNetwork = Model(physical_A_input,physical_A_output)
		
		optimization='Nadam'
		AlphaturnpropensityNetwork.compile(loss='binary_crossentropy',optimizer=optimization,metrics=[keras.metrics.binary_accuracy])
	else:
		AlphaturnpropensityNetwork = load_model("model/"+str(train_time-1)+'model/AlphaturnpropensityNetwork.h5')
	
	if(trainY is not None):
		#checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'model/AlphaturnpropensityNetwork.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min')
		weight_checkpointer = ModelCheckpoint(filepath="model/"+str(train_time)+'modelweight/AlphaturnpropensityNetworkweight.h5',verbose=1,save_best_only=True,monitor='val_loss',mode='min',save_weights_only=True)
		loss_checkpointer = LossModelCheckpoint(model_file_path="model/"+str(train_time)+'model/AlphaturnpropensityNetwork.h5',monitor_file_path="model/loss/"+str(train_time)+"Anetloss.json",verbose=1,save_best_only=True,monitor='val_loss',mode='min')		
		AfitHistory = AlphaturnpropensityNetwork.fit(train_physicalXa,trainY,batch_size=4096,nb_epoch=50,shuffle=True,callbacks=[early_stopping,loss_checkpointer,weight_checkpointer],class_weight='auto',validation_data=(val_physicalXa,valY))                  	
		AlphaturnpropensityNetwork.save("model/"+str(train_time)+'model/1AlphaturnpropensityNetwork.h5')
        
	return AlphaturnpropensityNetwork