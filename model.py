from transformers.models.beit.modeling_beit import BeitPreTrainedModel, BeitConfig, BeitModel
from transformers.modeling_outputs import DepthEstimatorOutput
from transformers import AutoImageProcessor
from torch import Tensor, nn
import torch

from typing import Optional

class SiLogLoss(nn.Module):
    r"""
    Implements the Scale-invariant log scale loss [Eigen et al., 2014](https://arxiv.org/abs/1406.2283).

    $$L=\frac{1}{n} \sum_{i} d_{i}^{2}-\frac{1}{2 n^{2}}\left(\sum_{i} d_{i}^{2}\right)$$ where $d_{i}=\log y_{i}-\log
    y_{i}^{*}$.

    """

    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

class BeitForDepthEstimation(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # FPNs
        if len(self.config.out_indices) != 4:
            raise ValueError(
                "BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers, "
                "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                "a base-sized architecture."
            )
        
        self.upscaleLayer = nn.Sequential(
            nn.ConvTranspose2d(3072, 3072//4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(3072//4, 3072//16, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(3072//16, 1, kernel_size=1, stride=1),
            nn.Tanh(),
        )
        # self.fpn1 = nn.Sequential(
        #     nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        #     nn.BatchNorm2d(config.hidden_size),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        # )
        # self.fpn2 = nn.Sequential(
        #     nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        # )
        # self.fpn3 = nn.Identity()
        # self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Semantic segmentation head(s)
        # self.decode_head = BeitUperHead(config)
        # self.auxiliary_head = BeitFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()  

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        batch_size = pixel_values.shape[0]
        patch_resolution = self.config.image_size // self.config.patch_size
        features = [
            x[:, 1:, :].permute(0, 2, 1).reshape(batch_size, -1, patch_resolution, patch_resolution) for x in features
        ]
        features = torch.cat(features, dim=1)
        # apply FPNs
        # ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        # for i in range(len(features)):
        #     features[i] = ops[i](features[i])
        loss = None
        predicted_depth = self.upscaleLayer(features)
        if labels is not None:
          siloss = SiLogLoss()
          loss = siloss(predicted_depth, labels)
        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions
        )




