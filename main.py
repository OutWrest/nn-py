from model import Model
from layer import Layer

model = Model()
model.addLayer(2, input_shape=2, )
model.addLayer(2)

weights = [
    [
        [
            .15, .20
        ],
        [
            .25, .30
        ]
    ],
    [
        [
            .40, .45
        ],
        [
            .50, .55
        ]
    ]
]

biases = [
    .35,
    .60
]

expected = [
    .01, .99
]

model.loadModel( weights, biases )

predicted = model.predict([.05, .10])
#print(model.local_cost(predicted, expected))
model.backpropagate([.05, .10], expected)