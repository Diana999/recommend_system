from algos.cpt import CPT
from algos.cpt_dumy import CPTDummy
from algos.cptfun import CPTFun
from implemnts.cpt_original import CPTMakeData


class GetCptResult:
    def __init__(self, style):
        self.model = None
        self.style = style
        self.choose_algo()
        self.bulild_data()
        self.train_model()
        self.predict_model()
        self.measure()

    def choose_algo(self):
        if self.style == 'dummy':
            self.model = CPTDummy()
        elif self.style == 'original':
            self.model = CPT()
        elif self.style == 'improved':
            self.model = CPTFun()  # или наоборт

    def bulild_data(self):
        data = CPTMakeData(num_of_seq=1000)  # splitting in 80/20 test/targer
        self.test_preffix, self.test, self.train = data.make_sequences()

    def train_model(self, merge=True):
        self.model.train(self.train, self.test, merge=merge)

    def predict_model(self):
        self.predictions, ttl = self.model.predict(self.train, self.test, 0.2, 10)

    def measure(self):
        supra = 0
        for ia in range(len(self.test)):
            if self.predictions[ia] and len(set(self.test_preffix[ia]) & set(self.predictions[ia])):
                supra += 1
        with open('shows_txt', 'a') as f:
            f.write(str(supra / len(self.predictions)))
