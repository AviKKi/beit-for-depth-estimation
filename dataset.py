import imgaug as ia
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision


class DIODEDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transformation_chain, min_depth, input_only=None):
        """
        Args:
            dataframe: pandas dataframe containing image path, depth map, and validity mask
            transformation_chain: composed chain of preprocessing transformations for data
            input_only (optional, str or List[str]): to denote the preprocessing layers to be discarded
                from `imgaug` transformation chain

        References:
            a. https://keras.io/examples/vision/depth_estimation/
            b. https://github.com/diode-dataset/diode-devkit/blob/master/diode.py
            c. https://github.com/fabioperez/pytorch-examples/blob/master/notebooks/PyTorch_Data_Augmentation_Image_Segmentation.ipynb
        """
        self.dataframe = dataframe
        self.transformation_chain = transformation_chain
        self.min_depth = min_depth
        self.input_only = input_only
        self.to_tensor = torchvision.transforms.ToTensor()

    def _process_depth_map(self, depth_map: np.ndarray, mask: np.ndarray):
        mask = mask > 0
        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, self.min_depth, np.log(max_depth))
        depth_map = np.expand_dims(depth_map, axis=2)
        return depth_map

    def _activator_masks(self, images, augmenter, parents, default):
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]["image"]
        depth_map_path = self.dataframe.iloc[idx]["depth"]
        mask_path = self.dataframe.iloc[idx]["mask"]

        image = PIL.Image.open(image_path).convert("RGB")
        image = np.asarray(image)

        depth_map = np.load(depth_map_path).squeeze()
        mask = np.load(mask_path)
        depth_map = self._process_depth_map(depth_map, mask)

        if self.input_only:
            det_tf = self.transformation_chain.to_deterministic()
            image = det_tf.augment_image(image)
            depth_map = det_tf.augment_image(
                depth_map, hooks=ia.HooksImages(activator=self._activator_masks)
            )
            image = self.to_tensor(image)
            depth_map = self.to_tensor(depth_map.copy())
        else:
            image = self.transformation_chain(image)
            depth_map = self.transformation_chain(depth_map)
        return {"image": image, "depth_map": depth_map}


def visualize_depth_map(samples, model=None):
    # Reference: https://keras.io/examples/vision/depth_estimation/#visualizing-samples
    input, target = samples["image"], samples["depth_map"]
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    if model:
        device = model.device
        inputs = {"pixel_values": input.to(device)}
        with torch.no_grad():
            outputs = model(**inputs).predicted_depth
            outputs = outputs.cpu().numpy()
            outputs = (outputs * 255 / np.max(outputs)).astype("uint8")

        fig, ax = plt.subplots(6, 3, figsize=(12, 20))
        for i in range(6):
            ax[i, 0].imshow(input[i].permute(1, 2, 0).numpy())
            ax[i, 1].imshow(target[i].permute(1, 2, 0).numpy().squeeze(), cmap=cmap)
            ax[i, 2].imshow(outputs[i].squeeze(), cmap=cmap)

    else:
        fig, ax = plt.subplots(6, 2, figsize=(8, 20))
        for i in range(6):
            ax[i, 0].imshow(input[i].permute(1, 2, 0).numpy().astype("float32"))
            ax[i, 1].imshow(target[i].permute(1, 2, 0).numpy().squeeze(), cmap=cmap)

    return fig