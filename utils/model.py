
# cfg.model

def get_model(cfg):
    if cfg.model == 'resnet18':
        from torchvision.models import resnet18
        return resnet18(num_classes=100)
    elif cfg.model == 'resnet34':
        from torchvision.models import resnet34
        return resnet34(num_classes=100)
    elif cfg.model == 'resnet50':
        from torchvision.models import resnet50
        return resnet50(num_classes=100)
    elif cfg.model == 'resnet101':
        from torchvision.models import resnet101
        return resnet101(num_classes=100)
    elif cfg.model == 'resnet152':
        from torchvision.models import resnet152
        return resnet152(num_classes=100)
    else:
        raise NotImplementedError(f"Model {cfg.model} is not implemented.")