from scripts.train.loss import BinaryFocalLoss, DiceLoss, SamLoss
from scripts.train.train import SamTrain, load_model

import torch


def load_batch_emb(path):
    return torch.load(path)


def test_focal_loss():
    batch_data: dict = load_batch_emb('batch_data.pt')
    k = list(batch_data.keys())[0]
    v = batch_data[k]

    loss = BinaryFocalLoss(gamma=2)

    target = torch.as_tensor(v['mask'] == 1).type(torch.int64)
    # m1 = masks.detach()[0, 0]
    good_loss: torch.Tensor = loss(input=target[None, ...], target=target)
    bad_loss: torch.Tensor = loss(input=1 - target[None, ...], target=target)
    assert (good_loss < bad_loss).all(), \
        "Good loss should be small"
    pass


def test_dice_loss():
    batch_data: dict = load_batch_emb('batch_data.pt')
    k = list(batch_data.keys())[0]
    v = batch_data[k]
    loss = DiceLoss(activation=None)
    target = torch.as_tensor(v['mask'] == 1).type(torch.int64)

    good_loss: torch.Tensor = loss(
        input=target[None, ...], target=target[None, ...]
    )
    bad_loss: torch.Tensor = loss(
        input=1 - target[None, ...], target=target[None, ...]
    )
    print(f"🚀 ~ file: test_loss.py:42 ~ good_loss: {good_loss} and {bad_loss}")


def test_sam_loss():
    batch_data: dict = load_batch_emb('batch_data.pt')
    k = list(batch_data.keys())[0]
    v = batch_data[k]

    sam = load_model()
    sam_train = SamTrain(sam_model=sam)

    masks_pred, iou_pred, _ = sam_train.predict_torch(
        image_emb=v['img_emb'],
        input_size=v['input_size'],
        original_size=v['original_size'],
        multimask_output=True,
        return_logits=True
    )

    loss = SamLoss()

    # 2, H, W
    target = torch.as_tensor(v['mask'] == 1).type(torch.int64)
    target = torch.stack((target, target))
    pred = torch.stack((masks_pred[0, 0, :, :], masks_pred[0, 0, :, :]))
    iou = torch.stack((iou_pred[0, 0], iou_pred[0, 0]))
    _l1 = loss(pred, target, iou)
    assert _l1[0] == _l1[1], \
        "Duplicate batch data should have the same loss"

    target = torch.as_tensor(v['mask'] == 1).type(torch.int64)[None, ...]
    pred = masks_pred[:, 0, :, :]
    iou = iou_pred[:, 0]
    _l2 = loss(pred, target, iou)
    assert _l2 == _l1[0], \
        "Loss from the same data must match"

    print("🚀 ~ file: test_loss.py:79 ~ _l:", _l1, _l2)

    pass


if __name__ == "__main__":
    test_focal_loss()
    test_dice_loss()
    test_sam_loss()

    pass