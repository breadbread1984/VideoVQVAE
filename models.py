#!/usr/bin/python3

import tensorflow as tf;

def AxialAttention(key_dim, value_dim, num_heads, drop_rate = 0.5, origin_shape = None, axial_dim = 0):
  # NOTE: this attention can only apply to self attention, but cross attention.
  # in other words, query_length = key_length must hold
  # NOTE: leave one dim as seq_length, merge the other dims with heads.
  # for example key.shape = (batch, heads, h, w, c, dim) and axial_dim = -2
  # key.shape becomes (batch, new_heads = heads * h * c, seq_length = w, dim),
  # the self attention matrix become w x w, rather than (h * w * c) x (h * w * c)
  assert type(origin_shape) in [list, tuple];
  assert 0 <= axial_dim < 3 or -3 <= axial_dim < 0;
  query = tf.keras.Input((num_heads, None, key_dim // num_heads)); # query.shape = (batch, heads, query_length, key_dim // heads)
  key = tf.keras.Input((num_heads, None, key_dim // num_heads)); # key.shape = (batch, heads, key_length, key_dim // heads)
  value = tf.keras.Input((num_heads, None, value_dim // num_heads)); # value.shape = (batch, heads, key_length, value_dim // heads)
  reshaped_query = tf.keras.layers.Reshape((num_heads, -1, origin_shape[0], origin_shape[1], key_dim // num_heads))(query);
  reshaped_key = tf.keras.layers.Reshape((num_heads, -1, origin_shape[0], origin_shape[1], key_dim // num_heads))(key);
  reshaped_value = tf.keras.layers.Reshape((num_heads, -1, origin_shape[0], origin_shape[1], value_dim // num_heads))(value);
  def get_perm(axial_dim):
    dims = np.arange(2 + 3 + 1); # batch x heads x *origin_shape x dim
    chosed_dim = 2 + axial_dim if axial_dim >= 0 else 2 + 3 + axial_dim;
    index = dims.tolist().index(chosed_dim);
    dims[index], dims[-2] = dims[-2], dims[index];
    return dims;
  reshaped_query = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(axial_dim)})(reshaped_query);
  reshaped_query = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_query); # query.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  reshaped_key = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(axial_dim)})(reshaped_key);
  reshaped_key = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_key); # key.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  reshaped_value = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_perm(axial_dim)})(reshaped_value);
  shape = tf.keras.layers.Lambda(lambda x: tf.shape(x))(reshaped_value);
  reshaped_value = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-2], tf.shape(x)[-1])))(reshaped_value); # value.shape = (batch, heads * np.prod(other_dims), axial_dim_length, dim)
  # 1) correlation matrix of query and key
  qk = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1], transpose_b = True))([reshaped_query, reshaped_key]);
  logits = tf.keras.layers.Lambda(lambda x, kd: x / tf.math.sqrt(tf.cast(kd, dtype = tf.float32)), arguments = {'kd': key_dim // num_heads})(qk); # logits.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  attention = tf.keras.layers.Softmax()(logits); # attention.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, key_length = axial_dim_length)
  attention = tf.keras.layers.Dropout(rate = drop_rate)(attention);
  # 2) weighted sum of value elements for each query element
  results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1]))([attention, reshaped_value]); # results.shape = (batch, heads * np.prod(other_dims), query_length = axial_dim_length, value_dim // heads)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], x[1]))([results, shape]); # results.shape = (batch, heads, *other_dims, axial_dim_length, value_dim // heads)
  def get_inv_perm(axial_dim):
    perm = get_perm(axial_dim);
    return np.argsort(perm);
  results = tf.keras.layers.Lambda(lambda x, p: tf.transpose(x, p), arguments = {'p': get_inv_perm(axial_dim)})(results); # results.shape = (batch, heads, *origin_shape, value_dim // heads)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1, tf.shape(x)[-1])))(results); # results.shape = (batch, heads, query_length = np.prod(origin_shape), value_dim // heads)
  return tf.keras.Model(inputs = (query, key, value), outputs = results);

def MultiHeadAttention(key_dim, value_dim, num_heads, **kwargs):
  query = tf.keras.Input((None, key_dim,)); # query.shape = (batch, query_length, key_dim)
  key = tf.keras.Input((None, key_dim,)); # key.shape = (batch, key_length, key_dim)
  value = tf.keras.Input((None, value_dim,)); # value.shape = (batch, key_length, value_dim)
  # 1) change to channels which can divided by num_heads
  query_dense = tf.keras.layers.Dense(units = key_dim // num_heads * num_heads)(query);
  key_dense = tf.keras.layers.Dense(units = key_dim // num_heads * num_heads)(key);
  value_dense = tf.keras.layers.Dense(units = value_dim // num_heads * num_heads)(value);
  # 2) split the dimension to form mulitiple heads
  query_splitted = tf.keras.layers.Reshape((-1, num_heads, key_dim // num_heads))(query_dense); # query_splitted.shape = (batch, query_length, num_heads, key_dim // num_heads)
  query_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(query_splitted); # query_splitted.shape = (batch, num_heads, query_length, key_dim // num_heads)
  key_splitted = tf.keras.layers.Reshape((-1, num_heads, key_dim // num_heads))(key_dense); # key_splitted.shape = (batch, key_length, num_heads, key_dim // num_heads)
  key_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(key_splitted); # key_splitted.shape = (batch, num_heads, key_length, key_dim // num_heads)
  value_splitted = tf.keras.layers.Reshape((-1, num_heads, value_dim // num_heads))(value_dense); # value_splitted.shape = (batch, key_length, num_heads, value_dim // num_heads)
  value_splitted = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(value_splitted); # value_splitted.shape = (batch, num_heads, key_length, value_dim // num_heads)
  attended = AxialAttention(key_dim, value_dim, num_heads, kwargs['drop_rate'], kwargs['origin_shape'], kwargs['axial_dim'])([query_splitted, key_splitted, value_splitted]); # reults.shape = (batch, num_heads, query_length, value_dim // num_heads)
  attended = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1, 3)))(attended); # attended.shape = (batch, query_length, num_heads, value_dim // num_heads)
  concated = tf.keras.layers.Reshape((-1, value_dim))(attended); # concated.shape = (batch, query_length, value_dim)
  # 3) output
  results = tf.keras.layers.Dense(key_dim)(concated); # results.shape = (batch, query_length, key_dim)
  return tf.keras.Model(inputs = (query, key, value), outputs = results);

def Conv3D(filters = None, kernel_size = None, strides = None, use_2d = False):
  assert type(kernel_size) in [list, tuple] and len(kernel_size) == 3;
  assert type(strides) in [list, tuple] and len(strides) == 3;
  def calc_pads():
    total_pad = [k - s for k,s in zip(kernel_size, strides)];
    pads = tuple([(p // 2 + p % 2, p // 2) for p in total_pad]);
    return pads;
  inputs = tf.keras.Input((None, None, None, None)); # inputs.shape = (batch, length, h, w, c)
  padded = tf.keras.layers.ZeroPadding3D(padding = calc_pads())(inputs);
  if use_2d == False:
    results = tf.keras.layers.Conv3D(filters, kernel_size, strides, padding = 'valid')(padded);
  else:
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4])))(padded); # results.shape = (batch * length, h, w, c)
    results = tf.keras.layers.Conv2D(filters, (kernel_size[1], kernel_size[2]), (strides[1], strides[2]), padding = 'valid')(results);
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], tf.shape(x[1])))([results, padded]); # results.shape = (batch, length, h, w, c)
    transposed = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 3, 1, 2, 4)))(results); # results.shape = (batch, w, length, h, c)
    
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4])))(transposed); # results.shape = (batch * w, length, h, c)
    results = tf.keras.layers.Conv2D(filters, (kernel_size[0], kernel_size[1]), (strides[0], strides[1]), padding = 'valid')(results);
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], tf.shape(x[1])))([results, transposed]); # results.shape = (batch, w, length, h, c)
    transposed = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 3, 1, 2, 4)))(results); # results.shape = (batch, h, w, length, c)
    
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3], tf.shape(x)[4])))(transposed); # results.shape = (batch * h, w, length, c)
    results = tf.keras.layers.Conv2D(filters, (kernel_size[2], kernel_size[0]), (strides[2], strides[0]), padding = 'valid')(results);
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], tf.shape(x[1])))([results, transposed]); # results.shape = (batch, h, w, length, c)
    transposed = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 3, 1, 2, 4)))(results); # results.shape = (batch, length, h, w, c)
    results = transposed;
  return tf.keras.Model(inputs = inputs, outputs = results);

def AttentionResidualBlock(in_channels, out_channels):
  inputs = tf.keras.Input((None, None, in_channels));
  short = inputs;
  results = tf.keras.layers.BatchNormalization()(inputs);
  results = tf.keras.layers.ReLU()(results);
  results = Conv3D(out_channels // 2, (3,3,3), (1,1,1))(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = Conv3D(out_channels, (1,1,1), (1,1,1))(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = 