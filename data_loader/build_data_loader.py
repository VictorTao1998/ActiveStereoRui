from torch.utils.data import DataLoader

from datasets.messytable import MessytableDataset
from datasets.messytable_test import MessytableTestDataset_TEST


def build_data_loader(cfg, mode="train"):

    if mode == "train":
        dataset = MessytableDataset(cfg.SPLIT.TRAIN, gaussian_blur=False, color_jitter=False, debug=False, sub=600)
    elif mode == "val":
        dataset = MessytableDataset(cfg.SPLIT.VAL, gaussian_blur=False, color_jitter=False, debug=False, sub=100, isVal=True)
    elif mode == "test":
        dataset = MessytableDataset(cfg.SPLIT.VAL, gaussian_blur=False, color_jitter=False, debug=False, sub=100, isVal=True)


    if mode == "train":
        batch_size = cfg.SOLVER.BATCH_SIZE
    else:
        batch_size = cfg.SOLVER.TEST_BATCH_SIZE

    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.SOLVER.NUM_WORKER,
        pin_memory=True,
    )

    return data_loader
