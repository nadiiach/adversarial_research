import plotly.graph_objects as go
import wandb

wandb.init(name='cross_domain', project='attack_benchmark')

figure = {
    "data": [
        {
            "meta": {
                "columnNames": {
                    "x": "A",
                    "y": "B"
                }
            },
            "name": "VGG16 Trained on Paintings",
            "type": "bar",
            "xsrc": "sauhaarda:10:e7fb4e",
            "x": [
                "VGG16",
                "VGG19",
                "ResNet 50",
                "ResNet 152",
                "Dense201",
                "SqueezeNet",
                "Inceptionv3"
            ],
            "ysrc": "sauhaarda:10:cc5fb8",
            "y": [
                "99.582%",
                "98.978%",
                "59.382%",
                "47.908%",
                "47.134%",
                "67.478%\n",
                "39.632%"
            ],
            "visible": True,
            "orientation": "v"
        },
        {
            "meta": {
                "columnNames": {
                    "x": "A",
                    "y": "C"
                }
            },
            "name": "ResNet152 Trained on Paintings",
            "type": "bar",
            "xsrc": "sauhaarda:10:e7fb4e",
            "x": [
                "VGG16",
                "VGG19",
                "ResNet 50",
                "ResNet 152",
                "Dense201",
                "SqueezeNet",
                "Inceptionv3"
            ],
            "ysrc": "sauhaarda:10:d612cb",
            "y": [
                "86.95%",
                "85.878%",
                "84.936%",
                "98.026%",
                "75.330%",
                "77.808%",
                "58.734%"
            ],
            "visible": True,
            "orientation": "v"
        }
    ],
    "layout": {
        "title": {
            "text": "Cross-Domain Attack Transfer Evaluation"
        },
        "xaxis": {
            "type": "category",
            "range": [
                -0.5,
                6.5
            ],
            "title": {
                "text": "Attacked Model (ImageNet Trained)"
            },
            "autorange": True,
            "showspikes": False
        },
        "yaxis": {
            "type": "linear",
            "range": [
                0,
                104.82315789473684
            ],
            "title": {
                "text": "Fooling Rate (%)"
            },
            "autorange": True,
            "showspikes": False
        },
        "legend": {
            "title": {
                "text": "<b>Pretrained Discriminator which<br>attack generator was trained with:</b>"
            }
        },
        "autosize": True,
        "annotations": []
    },
    "frames": []
}

wandb.log({'chart': go.Figure(figure)})
