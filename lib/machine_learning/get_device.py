import torch

def get_device() -> str:
    """
    Get device name.

    :return: String name of device
    """
    device = torch.accelerator.current_accelerator().type \
             if torch.accelerator.is_available() else 'cpu'
    print(f'Using {device} device')
    return device