from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class ConvAutoEncoder(Model):
  def __init__(self,
               input_shape,
               latent_dim = 2):
    super(ConvAutoEncoder, self).__init__()
  
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    encoder_layer_1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_layer_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(encoder_layer_1)
    encoder_layer_3 = layers.Flatten()(encoder_layer_2)
    encoder_layer_4 = layers.Dense(latent_dim,
                                   kernel_regularizer='l2')(encoder_layer_3)
    self.encoder = Model(encoder_input,
                         encoder_layer_4, 
                         name = 'encoder')

    # Decoder 
    decoder_input = layers.Input(shape = (latent_dim,))
    decoder_layer_1 = layers.Dense(units=32*1689*32,
                                   kernel_regularizer='l2')(decoder_input)
    decoder_layer_2 = layers.Reshape(target_shape=(32,1689,32))(decoder_layer_1)
    decoder_layer_3 = layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same')(decoder_layer_2)
    decoder_layer_4 = layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same')(decoder_layer_3)
    decoder_layer_5 = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(decoder_layer_4)
    self.decoder = Model(decoder_input, 
                         decoder_layer_5, 
                         name = 'decoder')

  def encode(self,x):
    return self.encoder(x)

  def decode(self, x):
    return self.decoder(x)

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded