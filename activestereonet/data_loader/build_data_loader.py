from torch.utils.data import DataLoader

from activestereonet.data_loader.sulab_indoor_active import SuLabIndoorActiveSet


def build_data_loader(cfg, mode="train"):
    if mode == "train":
        DATASET = cfg.DATA.TRAIN_DATASET
    elif mode == "val":
        DATASET = cfg.DATA.VAL_DATASET
    elif mode == "test":
        DATASET = cfg.DATA.TEST_DATASET
    else:
        raise ValueError("Unknown mode: {}.".format(mode))

    if DATASET == "SULAB_INDOOR_ACTIVE":
        if mode == "train":
            dataset = SuLabIndoorActiveSet(
                root_dir=cfg.DATA.TRAIN.ROOT_DIR,
                mode="train",
                view_list_file=cfg.DATA.TRAIN.VIEW_LIST_FILE,
                use_mask=cfg.DATA.TRAIN.USE_MASK,
            )
        elif mode == "val":
            dataset = SuLabIndoorActiveSet(
                root_dir=cfg.DATA.VAL.ROOT_DIR,
                mode="val",
                view_list_file=cfg.DATA.VAL.VIEW_LIST_FILE,
                use_mask=cfg.DATA.VAL.USE_MASK,
            )
        elif mode == "test":
            dataset = SuLabIndoorActiveSet(
                root_dir=cfg.DATA.TEST.ROOT_DIR,
                mode="test",
                view_list_file=cfg.DATA.TEST.VIEW_LIST_FILE,
                use_mask=cfg.DATA.TEST.USE_MASK,
            )

    # elif DATASET == "STD_ACTIVE":
    #     if mode == "train":
    #         dataset = StandardTestSetActive(
    #             root_dir=cfg.DATA.TRAIN.ROOT_DIR,
    #             mode="train",
    #             obj_list_file=cfg.DATA.TRAIN.OBJ_LIST_FILE,
    #             view_list_file=cfg.DATA.TRAIN.VIEW_LIST_FILE,
    #             min_depth=cfg.DATA.STD_MIN_DEPTH,
    #             max_depth=cfg.DATA.STD_MAX_DEPTH,
    #             num_view=cfg.DATA.TRAIN.NUM_VIEW,
    #             num_depth=cfg.DATA.TRAIN.NUM_DEPTH,
    #             use_mask=cfg.DATA.TRAIN.USE_MASK,
    #         )
    #     elif mode == "val":
    #         dataset = StandardTestSetActive(
    #             root_dir=cfg.DATA.VAL.ROOT_DIR,
    #             mode="val",
    #             obj_list_file=cfg.DATA.VAL.OBJ_LIST_FILE,
    #             view_list_file=cfg.DATA.VAL.VIEW_LIST_FILE,
    #             min_depth=cfg.DATA.STD_MIN_DEPTH,
    #             max_depth=cfg.DATA.STD_MAX_DEPTH,
    #             num_view=cfg.DATA.VAL.NUM_VIEW,
    #             num_depth=cfg.DATA.TRAIN.NUM_DEPTH,
    #             use_mask=cfg.DATA.VAL.USE_MASK,
    #         )
    #     elif mode == "test":
    #         dataset = StandardTestSetActive(
    #             root_dir=cfg.DATA.TEST.ROOT_DIR,
    #             mode="test",
    #             obj_list_file=cfg.DATA.TEST.OBJ_LIST_FILE,
    #             view_list_file=cfg.DATA.TEST.VIEW_LIST_FILE,
    #             min_depth=cfg.DATA.STD_MIN_DEPTH,
    #             max_depth=cfg.DATA.STD_MAX_DEPTH,
    #             num_view=cfg.DATA.TEST.NUM_VIEW,
    #             num_depth=cfg.DATA.TEST.NUM_DEPTH,
    #             use_mask=cfg.DATA.TEST.USE_MASK,
    #         )
    if mode == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True,
    )

    return data_loader
