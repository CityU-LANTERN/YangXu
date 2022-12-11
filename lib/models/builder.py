import yaml
import torch
import torchvision
import logging

logger = logging.getLogger()


def build_model(args, model_name, pretrained=False, pretrained_ckpt=''):
    assert model_name.startswith('timm_')
    import timm
    model = timm.create_model(model_name[5:], pretrained=pretrained, drop_path_rate=args.drop_path_rate, num_classes=args.num_classes)

    if pretrained and pretrained_ckpt != '':
        logger.info(f'Loading pretrained checkpoint from {pretrained_ckpt}')
        ckpt = torch.load(pretrained_ckpt, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']
        missing_keys, unexpected_keys = \
                model.load_state_dict(ckpt, strict=False)
        if len(missing_keys) != 0:
            logger.info(f'Missing keys in source state dict: {missing_keys}')
        if len(unexpected_keys) != 0:
            logger.info(f'Unexpected keys in source state dict: {unexpected_keys}')

    return model
