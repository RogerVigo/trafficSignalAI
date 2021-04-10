import sys
from model import TrainingModel, ProductionModel

if __name__ == "__main__":
    if 1 < len(sys.argv) > 1: #Estamos recibiendo parametros por cli
        if "-t" in sys.argv or "--train" in sys.argv:
            save = False
            if "-s" in sys.argv or "--save" in sys.argv: #Indicamos que queremos salvar el modelo
                save = True
            model = TrainingModel(save)
            model.run()

    elif len(sys.argv) == 1:
        try:
            model = ProductionModel()
        except (Exception):
            print("Error ocurred,", sys.exc_info())

        model.run()
