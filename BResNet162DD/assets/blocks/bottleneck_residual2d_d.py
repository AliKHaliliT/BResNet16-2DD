import tensorflow as tf
from ..layers.conv2d_layer import Conv2DLayer
from typing import Union, Any


@tf.keras.utils.register_keras_serializable()
class BottleneckResidual2DD(tf.keras.layers.Layer):

    """

    Bottleneck Residual2DD block from the paper "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    Link: https://arxiv.org/abs/1812.01187

    """

    def __init__(self, filters: int, strides: tuple[int, int], **kwargs) -> None:
        
        """

        Constructor of the BottleneckResidual2DD block.


        Parameters
        ----------
        filters : int
            Number of filters for the layers.

        strides : tuple
            Strides for the layers.

        
        Returns
        -------
        None.

        """

        super().__init__(**kwargs)

        self.filters = filters
        self.strides = strides

        # Main Path
        self.bottleneck = Conv2DLayer(filters=self.filters // 4 or 1, 
                                      kernel_size=(1, 1), 
                                      strides=(1, 1), 
                                      padding="same",
                                      use_bias=False, 
                                      normalization="batch_norm",
                                      activation="relu")
        self.feature_extract = Conv2DLayer(filters=self.filters // 4 or 1, 
                                           kernel_size=(3, 3), 
                                           strides=self.strides, 
                                           padding="same",
                                           use_bias=False,
                                           normalization="batch_norm",
                                           activation="relu")
        self.project = Conv2DLayer(filters=self.filters * 4, 
                                   kernel_size=(1, 1), 
                                   strides=(1, 1), 
                                   padding="same",
                                   use_bias=False, 
                                   normalization="batch_norm")

        # Residual Path
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=self.strides, padding="same")
        self.project1 = Conv2DLayer(filters=self.filters * 4, 
                                    kernel_size=(1, 1), 
                                    strides=(1, 1), 
                                    padding="same",
                                    use_bias=False, 
                                    normalization="batch_norm")

        # Finalize
        self.add = tf.keras.layers.Add()
        self.activate = tf.keras.layers.ReLU() 


    def build(self, input_shape: Union[tf.TensorShape, tuple[int, int, int, int]]) -> None:

        """

        Build method of the BottleneckResidual2DD block.

        
        Parameters
        ----------
        input_shape : tf.TensorShape or tuple
            Shape of the input tensor.
            

        Returns
        -------
        None.

        """

        super().build(input_shape)

        # Main Path
        self.bottleneck.build(input_shape)
        input_shape_transformed = self.bottleneck.compute_output_shape(input_shape)
        self.feature_extract.build(input_shape_transformed)
        input_shape_transformed = self.feature_extract.compute_output_shape(input_shape_transformed)
        self.project.build(input_shape_transformed)

        # Residual Path
        if input_shape != input_shape_transformed:
            self.project1.build(self.pool.compute_output_shape(input_shape))


    def call(self, X: tf.Tensor) -> tf.Tensor:

        """

        Call method of the BottleneckResidual2DD block.

        
        Parameters
        ----------
        X : tf.Tensor
            Input tensor.


        Returns
        -------
        output : tf.Tensor
            Output tensor.

        """

        # Main Path
        X_transformed = self.project(self.feature_extract(self.bottleneck(X)))

        # Residual Path
        if X.shape != X_transformed.shape:
            X = self.project1(self.pool(X))
            

        return self.activate(self.add([X, X_transformed]))
    

    def compute_output_shape(self, input_shape: Union[tf.TensorShape, tuple[int, int, int, int]]) -> tuple[int, int, int, int]:

        """

        Method to compute the output shape of the BottleneckResidual2DD block.

        
        Parameters
        ----------
        input_shape : tf.TensorShape or tuple
            Shape of the input tensor.

            
        Returns
        -------
        output_shape : tuple
            Shape of the output tensor.

        """

        # Main Path
        input_shape = self.bottleneck.compute_output_shape(input_shape)
        input_shape = self.feature_extract.compute_output_shape(input_shape)
        output_shape = self.project.compute_output_shape(input_shape)


        return output_shape
    

    def get_config(self) -> dict[str, Any]:

        """

         Method to get the configuration of the BottleneckResidual2DD block.


        Parameters
        ----------
        None.

        
        Returns
        -------
        config : dict
            Configuration of the BottleneckResidual2DD block.

        """

        config = super().get_config()

        config.update({
            "filters": self.filters,
            "strides": self.strides
        })


        return config
    

    def get_build_config(self) -> dict[str, Any]:

        """

         Method to get the build configuration of the BottleneckResidual2DD block.


        Parameters
        ----------
        None.

        
        Returns
        -------
        config : dict
            Configuration of the BottleneckResidual2DD block.

        """

        config = super().get_config()

        config.update({
            "filters": self.filters,
            "strides": self.strides
        })


        return config