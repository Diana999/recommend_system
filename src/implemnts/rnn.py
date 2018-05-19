from algos.neurals import BuildNeuralLSTM


def make_neural_work():
    smth = BuildNeuralLSTM()
    smth.make_data()
    smth.prepare_model()
    smth.fit_model()
    print(smth.gain_results())
