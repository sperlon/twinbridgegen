import tensorflow as tf
from tensorflow import keras


def CheckLSTMLayer(layer):
  n_args = len(layer)
  if n_args != 2:
    raise Exception(f"LSTM layer is incorrectly specified. The list must have exactly 2 arguments(LayerName:string, nWeights:int), but has {n_args}({layer}).")
  elif not isinstance(layer[1], int) and layer[1] > 0:
    raise Exception(f"Number of neurons must be non-zero non-negative integer, but you specified: {layer[1]}")


def CheckDenseLayer(layer):
  n_args = len(layer)
  if n_args == 2:
    layer.append(None)
    n_args += 1

  if n_args != 3:
    raise Exception(f"""Dense layer is incorrectly specified. The list should have 3 arguments(LayerName:string, nWeights:int, ActivationFunc:str), but has {n_args}({layer}).
     It is also possible to pass only 2 arguments(LayerName:string, nWeights:int), but in such case activation func would be None.""")

  elif not isinstance(layer[1], int) and layer[1] > 0:
    raise Exception(f"Number of neurons must be non-zero non-negative integer, but you specified: {layer[1]}")

  elif not isinstance(layer[2], str):
    if layer[2] is None:
        pass
    else:
      raise Exception(f"Activation function must be specified as string or none, but is {type(layer[0])}({layer[2]}) instead.")

def SubModel(layers, input):
  x = input
  n_layers = len(layers)

  first = True
  for i in range(n_layers):
    layer = layers[i]
    if layer[0].lower() == "lstm":
      CheckLSTMLayer(layer)
      # check whether the lstm layer is the last lstm in a row. If not return, full sequence
      if i == n_layers-1:
        ret_seq = False
      elif layers[i+1][0].lower() != "lstm":
        ret_seq = False
      else:
        ret_seq = True
      if first: # If LSTM is the first layer, it expects two-dimensional input. Hence we must manually reshape it
        x = tf.keras.layers.Reshape((x.shape[-1], 1))(x)
        first = False
      x = tf.keras.layers.LSTM(layer[1], return_sequences=ret_seq)(x)

    elif layer[0].lower() == "dense":
      CheckDenseLayer(layer)
      if first:
          first = False
      x = tf.keras.layers.Dense(layer[1], activation=layer[2])(x)

    else:
      raise Exception(f"Can not recognize layer: '{layer[0]}'.")

  return x


def MultiChannelModel(layers, input_dims, out_dim):
  inputs = [tf.keras.layers.Input(shape=[inp_dim]) for inp_dim in input_dims]
  sub_outputs = []

  for i in range(len(layers)):
    x = SubModel(layers[i], inputs[i])
    sub_outputs.append(x)

  x = tf.keras.layers.Add()(sub_outputs)
  out = tf.keras.layers.Dense(out_dim)(x)

  model = tf.keras.Model(inputs=inputs, outputs=out)
  return model

class MultiChannelModelC(tf.keras.Model):
    def __init__(self, layers=None, inp_dims=None, inp_slices=None, out_dim=None, **kwargs):
      super().__init__(**kwargs)

      # Store parameters for serialization
      self.layers_config = layers
      self.inp_dims = inp_dims
      self.inp_slices = inp_slices
      self.out_dim = out_dim

      # Only build the model if all parameters are provided
      if all(param is not None for param in [layers, inp_dims, inp_slices, out_dim]):
        self._build_model()

    def _build_model(self):
      """Build the internal model structure"""
      inputs = [tf.keras.layers.Input(shape=[inp_dim]) for inp_dim in self.inp_dims]
      sub_outputs = []

      for i in range(len(self.layers_config)):
        x = SubModel(self.layers_config[i], inputs[i])
        sub_outputs.append(x)

      x = tf.keras.layers.Add()(sub_outputs)
      out = tf.keras.layers.Dense(self.out_dim)(x)

      self.model = tf.keras.Model(inputs=inputs, outputs=out)

    def call(self, inputs):
      gathered_inputs = []

      for inp_slice in self.inp_slices:
        gathered_inputs.append(tf.gather(inputs, inp_slice, axis=-1))

      output = self.model(gathered_inputs)
      return output

    def get_config(self):
      """Return the configuration of the model for serialization"""
      config = super().get_config()
      config.update({
        'layers': self.layers_config,
        'inp_dims': self.inp_dims,
        'inp_slices': self.inp_slices,
        'out_dim': self.out_dim
      })
      return config

    @classmethod
    def from_config(cls, config):
      """Create an instance from config for deserialization"""
      return cls(**config)

    def build(self, input_shape):
      """Build the model when input shape is known"""
      if not hasattr(self, 'model') or self.model is None:
        self._build_model()
      super().build(input_shape)

    @property
    def input_shape(self):
      """Return the input shape of the model"""
      total_features = sum(self.inp_dims) if self.inp_dims else None
      return (None, total_features) if total_features else None

    @property
    def output_shape(self):
      """Return the output shape of the model"""
      return (None, self.out_dim) if self.out_dim else None


# Register the custom class for serialization (older TensorFlow/Keras versions)
tf.keras.utils.get_custom_objects()['MultiChannelModelC'] = MultiChannelModelC



class DenseModel(tf.keras.Model):
  def __init__(self, layers, inp_dim, out_dim):
    super().__init__()
    model = []
    model.append(tf.keras.layers.Input(shape=[inp_dim]))

    for layer in layers:
      model.append(tf.keras.layers.Dense(layer, activation="relu"))

    model.append(tf.keras.layers.Dense(out_dim))

    self.model = tf.keras.Sequential(model)

  def call(self, inputs):
    return self.model(inputs)


class DividedModel(tf.keras.Model):
  """
  Class for creation of a model where each member in output vector is predicted with different subnetwork.
  It turned out that for prediction of the strain from material parameters, it works better to have one smaller subnetwork
  for each strain than have one big network that the whole strain vector at once.

  You just have to specify layers in layers argument that each subnetwork will have (for example '[64, 64, 64]' would create
  subnetwork with 3 hidden layers with 64 neurons each), dimension of input vector(number of material params), and dimension
  of output vector(number of strains to predict).

  I designed this architecture specifically for the prediction of strain from material parameters, but if it proves advantageous,
  it can be used anywhere.

  Further there are methods for searching a corresponding input to given output (I want to find material parameters for
  specified strains). The newton´s method, the Gauss-newton´s method and SGD - stochastic gradient descent. However, for
  such a purpose I recommend to use only SGD, since the first two are unstable and most the time unable to converge
   """

  def __init__(self, layers=None, inp_dim=None, out_dim=None, act="relu", **kwargs):
    super().__init__(**kwargs)

    # Store parameters for serialization
    self.layers_config = layers
    self.inp_dim = inp_dim
    self.out_dim = out_dim
    self.act = act
    self.sub_models = []

    # Only build the model if all parameters are provided
    if all(param is not None for param in [layers, inp_dim, out_dim]):
      self._build_model()

  def _build_model(self):
    """Build the internal model structure"""
    # creation of submodels for each output parameter
    for o in range(self.out_dim):
      sub_model = tf.keras.Sequential(
        [tf.keras.layers.Dense(self.layers_config[0], activation=self.act, input_shape=[self.inp_dim])])
      for l in range(1, len(self.layers_config)):
        sub_model.add(tf.keras.layers.Dense(self.layers_config[l], activation=self.act))
      sub_model.add(tf.keras.layers.Dense(1))
      self.sub_models.append(sub_model)

  # keras method that needs to be defined. It specifies how output is calculated
  def call(self, inputs):
    part_outputs = []  # store each member of the output vector (strain) in a list
    for o in range(self.out_dim):
      part_outputs.append(
        self.sub_models[o](inputs))  # make prediction of output member (strain) by subnetwork and store it
    output = tf.keras.layers.Concatenate(axis=1)(part_outputs)  # concatenate the output to final output vector
    return output

  def get_config(self):
    """Return the configuration of the model for serialization"""
    config = super().get_config()
    config.update({
      'layers': self.layers_config,
      'inp_dim': self.inp_dim,
      'out_dim': self.out_dim,
      'act': self.act
    })
    return config

  @classmethod
  def from_config(cls, config):
    """Create an instance from config for deserialization"""
    return cls(**config)

  def build(self, input_shape):
    """Build the model when input shape is known"""
    if not hasattr(self, 'sub_models') or len(self.sub_models) == 0:
      self._build_model()
    super().build(input_shape)

  @property
  def input_shape(self):
    """Return the input shape of the model"""
    return (None, self.inp_dim) if self.inp_dim else None

  @property
  def output_shape(self):
    """Return the output shape of the model"""
    return (None, self.out_dim) if self.out_dim else None

  # SGD - stochastic gradient descent
  # ------------------------------------------------------------------------------------------------------------------
  # Getting the gradient with respect to input
  @tf.function  # this is a decorator that specifies for tensorflow to convert this method into the computational graph. As a result the computation is significantly faster and can run on GPU
  def get_grad_output_input(self, output, input):
    with tf.GradientTape() as tape:
      pred = self.call(input)
      l = tf.reduce_sum(tf.square(output - pred))

    grad = tape.gradient(l, input)
    return grad, l

  # This method search for optimal input for given output using gradient descent.
  # You can specify lower and upper limit for each parameter, tolerance (L2 norm between model prediction and searched output),
  # number of iterations and print_freq
  def find_input_SGD_based(self, output, input, optimizer, lower_limit=None, upper_limit=None, tolerance=1e-3,
                           max_iter=500, print_freq=100):
    out_ = tf.cast(output, tf.float32)
    inp_ = tf.cast(input, tf.float32)
    inp0 = tf.Variable(inp_, trainable=True)
    result = inp0.value()

    for i in range(max_iter):
      grad, l = self.get_grad_output_input(out_, inp0)
      optimizer.apply_gradients([(grad, inp0)])

      # check whether the parameters don´t exceed the limits
      if lower_limit is not None:
        if tf.reduce_any(inp0 < lower_limit):
          print("Lower limit broken")
          break

      elif upper_limit is not None:
        if tf.reduce_any(inp0 > upper_limit):
          print("Upper limit broken")
          break

      result = inp0.value()

      # print the loss value after the specified frequency, last iteration or when tolerance is achieved
      if i == 0 or i == max_iter - 1:
        print(f"Iteration {i}: loss = {l.numpy()}")
      elif (i + 1) % print_freq == 0:
        print(f"Iteration {i}: loss = {l.numpy()}")
      if l <= tolerance:
        print(f"Iteration {i}: loss = {l.numpy()}")
        break
    return result


# Register the custom class for serialization (older TensorFlow/Keras versions)
tf.keras.utils.get_custom_objects()['DividedModel'] = DividedModel
