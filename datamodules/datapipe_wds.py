import torch
import torchdata.datapipes as dp
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
import glob

def get_dataloader():

    # WebDataset 경로 설정 (샤드 파일 경로 패턴)
    shards1 = "/ssd0/data/cc3m/*.tar"
    shards2 = "/ssd0/data/cc12m/*.tar"

    # glob를 사용하여 파일 경로를 확장
    shard_files1 = glob.glob(shards1)
    shard_files2 = glob.glob(shards2)

    # 각 샤드에 대해 WebDataset 생성
    dataset1 = (
        wds.WebDataset(
            shard_files1,
            nodesplitter=wds.split_by_node,
            shardshuffle=True,
            detshuffle=False,
            resampled=False,
            handler=wds.ignore_and_continue,
        )
        .shuffle(size=1000, initial=100)  # 샘플 수준 셔플
        .decode("pil", handler=wds.ignore_and_continue)
        .to_tuple("jpg", "txt", handler=wds.ignore_and_continue)
    )

    dataset2 = (
        wds.WebDataset(
            shard_files2,
            nodesplitter=wds.split_by_node,
            shardshuffle=True,
            detshuffle=False,
            resampled=False,
            handler=wds.ignore_and_continue,
        )
        .shuffle(size=1000, initial=100)  # 샘플 수준 셔플
        .decode("pil", handler=wds.ignore_and_continue)
        .to_tuple("jpg", "txt", handler=wds.ignore_and_continue)
    )

    # SampleMultiplexer를 사용하여 여러 DataPipe를 하나로 결합
    datapipes = {dataset1: 1.0, dataset2: 1.0}
    combined_pipe = dp.iter.SampleMultiplexer(pipes_to_weights_dict=datapipes)

    # 이미지와 텍스트 데이터를 텐서로 변환하는 함수
    transform_function = transforms.ToTensor()
    tokenize_function = lambda x: x  # 예시로 identity 토크나이저 사용

    def transform_sample(sample):
        image, text = sample
        return transform_function(image), tokenize_function(text)

    # 데이터를 변환하는 DataPipe (예: 이미지 변환)
    transformed_data = combined_pipe.map(transform_sample)

    # 배치 처리를 위한 DataPipe
    batched_data = transformed_data.batch(batch_size=32)

    # collate_fn을 사용하여 리스트를 텐서로 변환
    def collate_fn(batch):
        images, texts = [], []
        for img, txt in batch[0]:
            images.append(img)
            texts.append(txt)
        return torch.stack(images), texts

    # DataLoader 생성
    dataloader = DataLoader(batched_data, num_workers=4, collate_fn=collate_fn)

    return dataloader