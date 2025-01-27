
dataset_type = 'LBAPDataset'
ann_file = '/data/home/wxl22/CSRL/data/lbap_core_ic50_scaffold.json'

train_pipeline = [
    dict(
        type="SmileToGraph",
        keys = ["input"]
    ),
    dict(
        type="Collect",
        keys=['input', 'gt_label', 'group']

    )
]

test_pipeline = [
    dict(
        type="SmileToGraph",
        keys = ["input"]
    ),
    dict(
        type="Collect",
        keys=['input', 'gt_label', 'group']
    )
]


data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(
        split="train",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline
    ),
    ood_val=dict(
        split="ood_val",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline
    ),
    ood_test=dict(
        split="ood_test",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
    ),
    iid_val = dict(
        split="iid_val",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline
    ),
    iid_test = dict(
        split="iid_val",
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline
    ),
    num_class=2
)