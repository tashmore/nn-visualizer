function feedForward(weights, input) {
    function activation(v) {
        return 1 / (1 + Math.exp(-v))
    } 

    values = [input]
    for (var fromLayer = 0; fromLayer < weights.length; fromLayer++) {
        values.push([])
        for (var toNode = 0; toNode < weights[fromLayer][0].length; toNode++) {
            values[fromLayer+1].push(0)
            for (var fromNode = 0; fromNode < weights[fromLayer].length; fromNode++) {
                values[fromLayer+1][toNode] += weights[fromLayer][fromNode][toNode] * values[fromLayer][fromNode]
            }
            values[fromLayer+1][toNode] = activation(values[fromLayer+1][toNode])
        }
        if (fromLayer < weights.length - 1)
            values[fromLayer+1].push(1)
    }
    return values
}

function inputGradient(fromLayer, fromNode, weights, values) {
    var deltas = []
    for (var layerIdx = values.length - 1; layerIdx >= 0; layerIdx--) {
        deltas.unshift([])
        var isOutputLayer = layerIdx == values.length - 1
        for (var nodeIdx = 0; nodeIdx < values[layerIdx].length; nodeIdx++) {
            var isBiasNode = (nodeIdx == values[layerIdx].length - 1) && !isOutputLayer
            var nodeValue = values[layerIdx][nodeIdx] * .9 + .1 // fudge so that (1 - nodeValue) * nodeValue doesn't vanish
            var delta
            if (isBiasNode) {
                delta = 0
            } else if (layerIdx == fromLayer && nodeIdx == fromNode) {
                //delta = (1 - nodeValue) * nodeValue * (1 - nodeValue)
                delta = 0.1
            } else if (layerIdx >= fromLayer) {
                delta = 0
            } else {
                delta = 0
                for (var wIdx = 0; wIdx < weights[layerIdx][nodeIdx].length; wIdx++) {
                    delta += weights[layerIdx][nodeIdx][wIdx] * deltas[1][wIdx]
                }
                delta *= nodeValue * (1 - nodeValue)
            }
            deltas[0].push(delta)
        }
    }
    return deltas[0]
}