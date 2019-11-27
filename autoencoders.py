from keras.layers import Input, Dense, Conv1D, Dropout, Flatten, Reshape, Add, Activation
from keras.models import Model
from keras import backend as K


class Causal1DAutoencoder():

    def __init__(self, input_shape, n_filters=64, latent_dims=10, dilation_rate=2, n_layers=4, n_residual_blocks=1, use_skip_connections=False):
        self.latent_dims = latent_dims
        self.input_shape = input_shape 
        self.n_filters = n_filters 
        self.dilation_rate = dilation_rate 
        self.n_layers = n_layers 
        self.n_residual_blocks = n_residual_blocks
        self.use_skip_connections = use_skip_connections
        self.autoencoder = None
        self.encoder = None
        self.decoder = None

    def build_model(self):
   
        input_ = Input(shape=self.input_shape)  

        # ENCODER
        encode = input_
        for _ in range(self.n_residual_blocks):
            encode = self._encoder_block(encode)
        encode = Flatten()(encode)
        encode = Dense(self.latent_dims)(encode)
        encode = Reshape((1,self.latent_dims))(encode)

        # DECODER
        decode = Dense(self.input_shape[0], activation='relu')(encode)
        decode = Reshape((self.input_shape[0],1))(decode)
        for _ in range(self.n_residual_blocks):
            decode = self._encoder_block(decode)
        decode = Conv1D(self.n_filters, 2, dilation_rate=1, activation='sigmoid', padding='causal')(decode)
        decode = Dense(self.input_shape[1])(decode)

        encoder = Model(input_, encode)

        autoencoder = Model(input_, decode)

        #autoencoder.compile(optimizer='adam', loss='mse')
        #encoder.compile()
        
        self.autoencoder = autoencoder
        self.encoder = encoder 
        self.decoder = 'Not implemented yet'


    def _block(self, x, dilation_steps):

        skip_connections = []

        for i in dilation_steps:
            x = Conv1D(self.n_filters, 2, dilation_rate=i, padding='causal')(x)
            skip_connections.append(x)
            x = Activation('relu')(x)

        if self.use_skip_connections == True:
            x = Add([x, skip_connections])
            return Activation('relu')(x)
        else:
            return x

    def _encoder_block(self, x):

        dilation_steps = [self.dilation_rate**j for j in range(self.n_layers)]
        return self._block(x, dilation_steps)

    def _decoder_block(self, x):

        dilation_steps = [self.dilation_rate**i for i in range(self.n_layers)][::-1][:-1]
        return self._block(x, dilation_steps)

    def compile_all(self, optimizer='adam', loss='mse'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        self.encoder.compile(optimizer=optimizer, loss=loss)



## Next TODO: write a model that clusters on the latent layer