
class BinaryConfig(object):
    # path to the train data file
    train_json = "data/intoxicated-speech-detection/dataset/Functionals_train_30_withBak.json"

    # path to the test/validation data file
    test_json = "data/intoxicated-speech-detection/dataset/Functionals_valid_30_withBak.json"

    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    HIDDEN_UNITS = 100
    LEARNING_RATE = 0.0001



class Multiclass(object):
     # path to the train data file
    train_json = "data/intoxicated-speech-detection/dataset/Functionals_train_30_withBak.json"

    # path to the test/validation data file
    test_json = "data/intoxicated-speech-detection/dataset/Functionals_valid_30_withBak.json"

    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    HIDDEN_UNITS = 100
    LEARNING_RATE = 0.0001
