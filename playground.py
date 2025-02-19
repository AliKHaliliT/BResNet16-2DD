import os
# Reduce TensorFlow log level for minimal logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'


from BResNet162DD import BResNet162DD


model = BResNet162DD()
model.build((None, 256, 256, 3))
model.summary()