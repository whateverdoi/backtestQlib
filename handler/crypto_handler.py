from qlib.contrib.data.handler import Alpha158


class CryptoAlphaHandler(Alpha158):
    def get_label_config(self):
        return ["Ref($label, -1)"], ["LABEL0"]

    def __init__(self, **kwargs):
        kwargs.setdefault("infer_processors", [{"class": "Fillna", "kwargs": {}}])
        kwargs.setdefault("learn_processors", [{"class": "DropnaLabel"}])
        super().__init__(**kwargs)
