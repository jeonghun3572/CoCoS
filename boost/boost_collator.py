import warnings
import numpy as np

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

class BoostCollator(DataCollatorForLanguageModeling):
    # Train only on tokens after the last [CORRECT (second turn), or after the last [BEGIN (first turn)
    def __init__(
        self,
        response_template: Union[str, list[int]],
        response_template_2: Union[str, list[int]],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.response_token_ids = self._to_ids(response_template)
        self.response_token_ids_2 = self._to_ids(response_template_2)
        self.ignore_index = ignore_index

    def _to_ids(self, template):
        if isinstance(template, str):
            return self.tokenizer.encode(template, add_special_tokens=False)
        return template

    def _find_last(self, labels, token_ids):
        start_idx = None
        for idx in np.where(labels == token_ids[0])[0]:
            if token_ids == labels[idx : idx + len(token_ids)].tolist():
                start_idx = idx
        return start_idx

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_start_idx = self._find_last(batch["labels"][i], self.response_token_ids_2)
            if response_token_ids_start_idx is not None:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids_2)
                batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index
                continue

            response_token_ids_start_idx = self._find_last(batch["labels"][i], self.response_token_ids)
            if response_token_ids_start_idx is None:
                warnings.warn(
                    f"Could not find `{self.tokenizer.decode(self.response_token_ids)}` {self.response_token_ids} or "
                    f"`{self.tokenizer.decode(self.response_token_ids_2)}` {self.response_token_ids_2} in the following instance: "
                    f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss calculation. "
                    "Make sure the data follows the Boost data format with [BEGIN] and [CORRECT] (see README). "
                    "If the data is correct, the instance may have been truncated by `max_seq_len`.",
                    UserWarning,
                )
                batch["labels"][i, :] = self.ignore_index
            else:
                response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
                batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        return batch
