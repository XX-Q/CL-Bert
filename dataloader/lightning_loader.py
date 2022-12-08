import pytorch_lightning as pl
import torch

from .ChIDDatasetIC import ChIDDatasetIC
from .ChIDDatasetIE import ChIDDatasetIE


class ChIDDataLoader(pl.LightningDataModule):
    def __init__(self,
                 data_path,
                 chid_file,
                 batch_size,
                 tokenizer_name,
                 task_type,
                 max_length=300,
                 idiom_mask_length=4,
                 replace_idiom=False,
                 num_workers=8):

        super().__init__()
        self._data_path = data_path
        self._chid_file = chid_file
        self._batch_size = batch_size
        self._tokenizer_name = tokenizer_name
        self._max_length = max_length
        self._num_workers = num_workers
        self._idiom_mask_length = idiom_mask_length
        self._replace_idiom = replace_idiom
        self._task_type = task_type
        if self._task_type == "IE":
            dataset_model = ChIDDatasetIE
        elif self._task_type == "IC":
            dataset_model = ChIDDatasetIC
        else:
            raise ValueError("task_type must be IE or IC")

        self.train_dataset = dataset_model(self._data_path,
                                           chid_file=self._chid_file,
                                           tokenizer_name=self._tokenizer_name,
                                           max_len=self._max_length,
                                           idiom_mask_length=self._idiom_mask_length,
                                           replace_idiom=self._replace_idiom, )
        self.val_dataset = dataset_model(self._data_path,
                                         chid_file="dev_data.json",
                                         tokenizer_name=self._tokenizer_name,
                                         max_len=self._max_length,
                                         idiom_mask_length=self._idiom_mask_length,
                                         replace_idiom=self._replace_idiom, )

        self.test_dataset = dataset_model(self._data_path,
                                          chid_file="test_data.json",
                                          tokenizer_name=self._tokenizer_name,
                                          max_len=self._max_length,
                                          idiom_mask_length=self._idiom_mask_length,
                                          replace_idiom=self._replace_idiom, )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=True,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           drop_last=False)
