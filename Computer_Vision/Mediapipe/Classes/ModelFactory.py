import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

class ModelFactory:

    @staticmethod
    def mlp_basic(input_shape, class_count):
        """
        Basic Multi-Layer Perceptron (MLP) - Baseline model from previous team.
        """
        return models.Sequential([
            layers.Dense(512, activation='relu', input_shape=[input_shape]),
            layers.Dense(256, activation='relu'),
            layers.Dense(class_count, activation='softmax')
        ])

    @staticmethod
    def mlp_with_dropout(input_shape, class_count):
        """
        Regularised MLP - Baseline model with L2 Regularisation and Dropout.
        Regularisation reduces variance in model predictions and aim for better generalisation.
        """
        return models.Sequential([
            layers.Dense(512, activation='relu', input_shape=[input_shape],
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu',
                         kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(class_count, activation='softmax')
        ])

    @staticmethod
    def mlp_deep(input_shape, class_count):
        """
        Deeper MLP with batch normalisation.
        5 Hidden layers instead of 2.
        Batch normalisation can stabilise training by normalising layer inputs.
        """
        return models.Sequential([
            layers.Dense(1024, activation='relu', input_shape=[input_shape]),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(class_count, activation='softmax')
        ])

    @staticmethod
    def mlp_attention(input_shape, class_count):
        """
        MLP with self-attention mechanism.
        Uses multi-head attention to learn relationships between different coordinates.
        (i.e. Helps to identify which joints or features are most important per pose)
        """
        
        # Define the input layer
        inputs = layers.Input(shape=input_shape)  # (33, 4)
        
        # Query, key, and value all come from the same input tensor in this case
        query = inputs
        key = inputs
        value = inputs
        
        # Apply MultiHeadAttention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=1,
            key_dim=32
        )(query, key, value)        

        # Continue the model as usual
        x = layers.Flatten()(attention_output)  # Keep all attention info instead of averaging
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(class_count, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    

    @staticmethod
    def cnn_basic(input_shape, class_count):
        """
        1D Convolutional Neural Network (CNN).
        Slides filters (size 3) across features (stride 1) to detect patterns.
        """
        return models.Sequential([
            layers.Input(shape=input_shape),  # (33, 4)
            
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.GlobalMaxPooling1D(),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(class_count, activation='softmax')
        ])

    @staticmethod
    def cnn_attention(input_shape, class_count):
        """
        1D CNN with attention mechanism for enhanced feature selection.
        """
        inputs = layers.Input(shape=input_shape)  # (33, 4)

        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)

        # Attention portion
        attention = layers.Dense(1, activation='tanh')(x)     # (batch, 16, 1)
        attention = layers.Softmax(axis=1)(attention)
        x = layers.Multiply()([x, attention])

        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(class_count, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs)
    
    @staticmethod
    def cnn_2d(input_shape, class_count):
        """
        2D CNN treating data as 2D image.
        Converts 1D sequence (33, 4) to 2D representation (33, 4, 1) for spatial convolutions.
        """
        inputs = layers.Input(shape=input_shape)  # (33, 4)
        x = layers.Reshape((33, 4, 1))(inputs)

        x = layers.Conv2D(32, (3, 2), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 2), activation='relu', padding='same')(x)
        x = layers.GlobalMaxPooling2D()(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(class_count, activation='softmax')(x)

        return tf.keras.Model(inputs, outputs)