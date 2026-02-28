from tensorflow.keras.layers import Layer, Conv3D, Activation
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LayerNormalization, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate, Dropout, Activation, add, UpSampling3D, AveragePooling3D, Lambda, subtract, multiply
from tensorflow.keras.layers import Conv3D, Activation
import tensorflow as tf

class DepthwiseSeparableConv3D(Layer):
    def __init__(self, filters, kernel_size=1, strides=1, dilation_rate=1,
                 padding='same', use_bias=True, activation='relu', **kwargs):
        super(DepthwiseSeparableConv3D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation

    def build(self, input_shape):
        in_channels = input_shape[-1]

        # Depthwise conv: groups = in_channels
        self.depthwise_conv = Conv3D(
            filters=in_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
            groups=in_channels,
            use_bias=self.use_bias
        )

        # Pointwise conv: 1x1x1
        self.pointwise_conv = Conv3D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            use_bias=self.use_bias
        )

        # Activation (if provided)
        if self.activation:
            self.act = Activation(self.activation)
        else:
            self.act = None

        super(DepthwiseSeparableConv3D, self).build(input_shape)

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)
        if self.act:
            x = self.act(x)
        return x

# Register the custom layer globally
get_custom_objects().update({'DepthwiseSeparableConv3D': DepthwiseSeparableConv3D})




def multi_scale_context_modulation(x, filters, dilation_rate, use_1x1_fusion, activation):
    # Parallel depthwise separable convolutions with different dilation rates
    convs = [
        DepthwiseSeparableConv3D(filters=filters, kernel_size=1, strides=1,
                                 dilation_rate=d, activation=activation,
                                 padding='same', use_bias=True)(x)
        for d in dilation_rate
    ]
    
    fusion = add(convs)
    
    if use_1x1_fusion:
        fusion = DepthwiseSeparableConv3D(filters=filters, kernel_size=1, activation=activation)(fusion)

    # Attention gating
    theta_x = Conv3D(filters // 2, 1, padding='same')(x)
    phi_g = Conv3D(filters // 2, 1, padding='same')(fusion)
    add_xg = Activation('relu')(add([theta_x, phi_g]))
    psi = Conv3D(1, 1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)
    
    # Final output modulated by attention
    fusion = multiply([fusion, psi])
    return fusion

# --- Hybrid Frequency Feature Extraction ---
def hybrid_frequency_feature_extraction(x, filters, pooling_type, combine_strategy, activation, kernel_size):
    low = AveragePooling3D(pool_size=2)(x) if pooling_type == 'avg' else MaxPooling3D(pool_size=2)(x)
    
    # Apply DepthwiseSeparableConv3D properly
    low = DepthwiseSeparableConv3D(filters=filters, kernel_size=1, activation=activation)(low)
    low = UpSampling3D(size=2)(low)
    
    high = subtract([x, low])
    high = DepthwiseSeparableConv3D(filters=filters, kernel_size=1, activation=activation)(high)

    combined = add([low, high]) if combine_strategy == 'add' else concatenate([low, high], axis=-1)
    return Activation(activation)(combined)

# --- Uncertainty-Guided Feature Refinement ---
def uncertainty_guided_refinement(x, filters, uncertainty_weight, activation, kernel_size):
    var = Lambda(
        lambda x: tf.reduce_mean(
            tf.square(x - tf.reduce_mean(x, axis=-1, keepdims=True)),
            axis=-1, keepdims=True)
    )(x)

    mask = Activation('relu')(Conv3D(1, 1, padding='same')(var))
    refined = multiply([x, mask])
    conv = DepthwiseSeparableConv3D(filters=filters, kernel_size=1, activation=activation)(refined)

    return Lambda(lambda t: t * uncertainty_weight)(conv)


# --- U-Net Model ---
def build_unet_model(input_shape, num_layers, filters, activation, dropout_rate, num_classes,
                    kernel_size, pooling_type, combine_strategy, dilation_rate, use_1x1_fusion,
                    uncertainty_weight):
    inputs = Input(shape=input_shape)
    x = Conv3D(filters, kernel_size=1, activation=activation, padding='same')(inputs)
    x = LayerNormalization()(x)

    # x = inputs
    encoder_layers = []

    
    # Encoder
    current_filters = filters
    for _ in range(num_layers):
        x = Conv3D(current_filters, kernel_size = 1, activation=activation, padding='same')(x)
        x = hybrid_frequency_feature_extraction(x, current_filters, pooling_type, combine_strategy, activation, kernel_size)
        encoder_layers.append(x)
        x = MaxPooling3D(pool_size=2)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        current_filters *= 2

    # Middle
    x = Conv3D(current_filters, kernel_size=5, activation=activation, padding='same')(x)
    x = uncertainty_guided_refinement(x, current_filters, uncertainty_weight, activation, kernel_size)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    previous_decoder_output = None

    # Decoder
    for _ in range(num_layers):
        current_filters //= 2
        x = Conv3DTranspose(current_filters, kernel_size=3, strides=2, padding='same')(x)
        x = Activation(activation)(x)

        skip = encoder_layers.pop()
        skip = multi_scale_context_modulation(skip, current_filters, dilation_rate, use_1x1_fusion, activation)

        # --- Apply attention gate ---
        if previous_decoder_output is not None:
            prev_up = UpSampling3D(size=2)(previous_decoder_output)
            x = concatenate([x, skip, prev_up])
        else:
            x = concatenate([x, skip])

        x = Conv3D(current_filters, kernel_size=1, activation=activation, padding='same')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

        previous_decoder_output = x

    # Output
    outputs = Conv3D(num_classes, 1, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model



