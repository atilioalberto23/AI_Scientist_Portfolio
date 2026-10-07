import torch.nn as nn


class TinyNet(nn.Module):
    """
    Baseline neural network used in Gradient Clinic.

    Architecture:
        input
        -> Linear(20, 64)
        -> BatchNorm
        -> ReLU
        -> Dropout
        -> Linear(64, 32)
        -> ReLU
        -> Dropout
        -> Linear(32, 1)

    The final layer returns a logit, not a probability.
    """

    def __init__(
        self,
        n_features,
        hidden_1=64,
        hidden_2=32,
        dropout_1=0.20,
        dropout_2=0.10
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                n_features,
                hidden_1
            ),

            nn.BatchNorm1d(
                hidden_1
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_1
            ),

            nn.Linear(
                hidden_1,
                hidden_2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_2
            ),

            nn.Linear(
                hidden_2,
                1
            )
        )

    def forward(self, x):

        return self.network(x).squeeze(1)


class MLPClassifier(nn.Module):

    def __init__(
        self,
        n_features,
        hidden_layers,
        dropout=0.0,
        batch_norm=True
    ):
        super().__init__()

        layers = []

        input_size = n_features

        for hidden_size in hidden_layers:

            layers.append(
                nn.Linear(
                    input_size,
                    hidden_size
                )
            )

            if batch_norm:
                layers.append(
                    nn.BatchNorm1d(
                        hidden_size
                    )
                )

            layers.append(
                nn.ReLU()
            )

            if dropout > 0:
                layers.append(
                    nn.Dropout(dropout)
                )

            input_size = hidden_size

        layers.append(
            nn.Linear(
                input_size,
                1
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.network(x).squeeze(1)