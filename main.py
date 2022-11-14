from engine import Engine
from animegan import AnimeGAN

if __name__ == '__main__':
    animegan = AnimeGAN("models/Hayao_64.onnx")
    engine = Engine(webcam_id=0, show=True, custom_objects=[animegan])
    engine.run()