import torch
import torch.nn as nn
import torch.nn.functional as F


def local_contrast_norm(image, kernel_size=9, eps=1e-5):
    """compute local contrast normalization
    input:
        image: torch.tensor (batch_size, 1, height, width)
    output:
        normed_image
    """
    assert (kernel_size % 2 == 1), "Kernel size should be odd"
    batch_size, channel, height, width = image.shape
    assert (channel == 1), "Only support single channel image for now"
    unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1) // 2)
    unfold_image = unfold(image)  # (batch, kernel_size*kernel_size, height*width)
    avg = torch.mean(unfold_image, dim=1).contiguous().view(batch_size, 1, height, width)
    std = torch.std(unfold_image, dim=1, unbiased=False).contiguous().view(batch_size, 1, height, width)

    normed_image = (image - avg) / (std + eps)

    return normed_image, std


class Fetch_Module(nn.Module):
    def __init__(self, padding_mode="zeros"):
        super(Fetch_Module, self).__init__()
        self.padding_mode = padding_mode

    def forward(self, right_img, disp):
        assert disp.shape == right_img.shape
        batch_size, channel, height, width = right_img.shape

        x_grid = torch.linspace(0., width - 1, width, dtype=disp.dtype, device=disp.device)\
            .view(1, 1, width, 1).expand((batch_size, height, width, 1))
        y_grid = torch.linspace(0., height - 1, height, dtype=disp.dtype, device=disp.device)\
            .view(1, height, 1, 1).expand((batch_size, height, width, 1))

        x_grid = x_grid - disp.permute(0, 2, 3, 1)
        x_grid = (x_grid - (width - 1) / 2) / (width - 1) * 2
        y_grid = (y_grid - (height - 1) / 2) / (height - 1) * 2
        xy_grid = torch.cat([x_grid, y_grid], dim=-1)

        reproj_img = F.grid_sample(right_img, xy_grid, padding_mode=self.padding_mode)

        return reproj_img


class Windowed_Matching_Loss(nn.Module):
    def __init__(self, lcn_kernel_size=9, window_size=33, sigma_weight=2,
                 invalid_reg_weight=1.0, invalid_weight=1.0):
        super(Windowed_Matching_Loss, self).__init__()
        self.lcn_kernel_size = lcn_kernel_size
        self.window_size = window_size
        self.sigma_weight = sigma_weight
        self.invalid_reg_weight = invalid_reg_weight
        self.invalid_weight = invalid_weight
        self.fetch_module = Fetch_Module()

    def forward(self, preds, data_batch):
        left_ir, right_ir = data_batch["left_ir"], data_batch["right_ir"]
        batch_size, channel, height, width = left_ir.shape
        assert (channel == 1)
        lcn_left_ir, left_std = local_contrast_norm(left_ir, self.lcn_kernel_size)
        lcn_right_ir, _ = local_contrast_norm(right_ir, self.lcn_kernel_size)

        disp = preds["refined_disp"]
        reproj_ir = self.fetch_module(lcn_right_ir, disp)
        reconstruct_err = F.l1_loss(lcn_left_ir, reproj_ir, reduction='none')
        C = left_std * reconstruct_err
        if self.window_size != 1:
            assert (self.window_size % 2 == 1)
            unfold = nn.Unfold(self.window_size, padding=(self.window_size - 1) // 2)
            C = unfold(C)  # (batch_size, window_size*window_size, height*width)
            Ixy = unfold(lcn_left_ir)
            wxy = torch.abs(lcn_left_ir.view(batch_size, 1, -1) - Ixy)
            wxy = torch.exp(-wxy / self.sigma_weight)
            C = (C * wxy).sum(1) / (wxy.sum(1))

        losses = {}
        if "invalid_mask" in preds.keys():
            invalid_mask = preds["invalid_mask"]
            invalid_reg_loss = (- torch.log(1 - invalid_mask)).mean()
            losses["invalid_reg_loss"] = invalid_reg_loss * self.invalid_reg_weight
            rec_loss = (C * (1 - invalid_mask)).mean()
            losses["rec_loss"] = rec_loss

            right_disp = preds["right_disp"]
            reproj_disp = self.fetch_module(right_disp, disp)

            disp_consistency = torch.abs(disp - reproj_disp)
            invalid_mask_from_consistency = (disp_consistency > 1).float()
            invalid_loss = (-torch.log(invalid_mask) * invalid_mask_from_consistency
                            - torch.log(1 - invalid_mask) * (1 - invalid_mask_from_consistency)).mean()
            losses["invalid_loss"] = invalid_loss * self.invalid_weight
        else:
            losses["rec_loss"] = C.mean()

        return losses


class Supervision_Loss(nn.Module):
    def __init__(self, invalid_weight=1.0):
        super(Supervision_Loss, self).__init__()
        self.invalid_weight = invalid_weight

    def forward(self, preds, data_batch):
        disp_pred = preds["refined_disp"]
        disp_gt = data_batch["disp_map"]

        invalid_mask_pred = preds["invalid_mask"]
        invalid_mask_gt = data_batch["invalid_mask"]
        valid_mask_gt = 1 - invalid_mask_gt

        invalid_loss = -(torch.log(invalid_mask_pred) * invalid_mask_gt + torch.log(1 - invalid_mask_pred) * (
                    1 - invalid_mask_gt)).mean()

        disp_loss = (torch.abs(disp_pred - disp_gt) * valid_mask_gt).sum() / (valid_mask_gt.sum() + 1e-7)

        return {
            "disp_loss": disp_loss,
            "invalid_loss": invalid_loss * self.invalid_weight,
        }


def test_lcn(image_path):
    import cv2
    source_image = cv2.imread(image_path, 0)
    source_image_tensor = torch.tensor(source_image).float().unsqueeze(0).unsqueeze(0)
    normed_image = local_contrast_norm(source_image_tensor)
    normed_image = normed_image.squeeze().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(normed_image)
    plt.show()


def test_fetch():
    batch_size = 2
    height, width = 240, 320
    disp = 10

    left_image = torch.rand((batch_size, 1, height, width)).float()
    pad = torch.zeros((batch_size, 1, height, disp)).float()
    disp_map = torch.ones((batch_size, 1, height, width + disp)).float() * disp

    right_image = torch.cat([left_image, pad], dim=-1)

    fetch_module = Fetch_Module()
    reproj_image = fetch_module(right_image, disp_map)[:, :, :, disp:]

    print(torch.allclose(left_image, reproj_image, atol=1e-4))


if __name__ == '__main__':
    # test_lcn("/media/rayc/文档/我的坚果云/SU Lab/sofa_0/coded_light/0.png")
    test_fetch()
