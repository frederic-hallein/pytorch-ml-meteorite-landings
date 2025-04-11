import torch

def save_model(model: any, model_path: str) -> None:
    """
    Save model.

    :param model: NeuralNetwork model
    :param model_path: String name of model path
    :return: None
    """
    torch.save(model.state_dict(), model_path)
    print(f"Saved model state to '{model_path}'")