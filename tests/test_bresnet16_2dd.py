import unittest
from BResNet162DD.bresnet16_2dd import BResNet162DD
import tensorflow as tf
import inspect


class TestBResNet162DD(unittest.TestCase):

    def test_units_wrong__value_value__error(self):

        # Arrange
        units = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            BResNet162DD(units=units)


    def test_creation_init_model(self):

        # Arrange and Act
        model = BResNet162DD()

        # Assert
        self.assertTrue(isinstance(model, tf.keras.Model))


    def test_build_input__shape_model(self):

        # Arrange
        model = BResNet162DD()
        input_shape = (None, 32, 32, 3)

        # Act
        model.build(input_shape=input_shape)

        # Assert
        self.assertTrue(model.built)


    def test_compile_setup_model(self):

        # Arrange
        model = BResNet162DD()

        # Act
        model.compile(optimizer="adam", loss="mse")

        # Assert
        self.assertTrue(model.optimizer is not None)


    def test_fit_input_history(self):

        # Arrange
        X = tf.random.uniform((1, 32, 32, 3))
        y = tf.random.uniform((1, 1))
        model = BResNet162DD()

        # Act
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(X, y, epochs=2)

        # Assert
        self.assertIsInstance(history, tf.keras.callbacks.History)


    def test_get__config_init_matching__dict(self):

        # Arrange
        model = BResNet162DD()

        # Act
        init_params = [
            param.name
            for param in inspect.signature(BResNet162DD.__init__).parameters.values()
            if param.name != "self" and param.name != "kwargs" 
        ]

        # Assert
        self.assertTrue(all(param in model.get_config() for param in init_params), "Missing parameters in get_config.")


    def test_from__config_config__dict_inputted__config__dict(self):

        # Arrange
        model = BResNet162DD()

        # Act
        cloned = BResNet162DD.from_config(model.get_config())

        # Assert
        self.assertEqual(model.get_config(), cloned.get_config())


if __name__ == "__main__":
    unittest.main()