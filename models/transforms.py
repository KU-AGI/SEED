from torchvision import transforms


def get_transform(type='clip', keep_ratio=True, image_size=224, normalize=True):
    if type == 'clip':
        transform = []
        if keep_ratio:
            transform.extend([
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
            ])
        else:
            transform.append(transforms.Resize((image_size, image_size), antialias=True))
        transform.extend([
            transforms.ToTensor(),
        ])
        if normalize:
            transform.append(
                transforms.Normalize(mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989))
            )

        return transforms.Compose(transform)
    else:
        raise NotImplementedError
