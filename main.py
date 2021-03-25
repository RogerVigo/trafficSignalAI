import sys
from model import TrainingModel as tm

if __name__ == "__main__":
    if 1 < len(sys.argv) < 3: #Estamos recibiendo parametros por cli
        if sys.argv[1] == "-t" or sys.argv[1] == "--train":
            model = tm()
            model.run()
            pass #si hay que entrenar el modelo, hacer algo

    elif len(sys.argv) == 1:
        pass #No hay parametros, hacer lo que haya que hacer