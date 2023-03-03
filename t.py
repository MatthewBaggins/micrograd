from micrograd.value import Value
from micrograd.mlp import MLP
from micrograd.utils import flatten

def main() -> None:
    xs = [
        Value.make(2, 3, -1),
        Value.make(3, -1, 0.5),
        Value.make(0.5, 1, 1),
        Value.make(1, 1, -1),
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    net = MLP(3, 2, 4, 1)
    n_epochs = 1000
    for epoch in range(n_epochs):        
        preds = flatten(net.batch_call(xs))
        loss = net.compute_loss(ys, preds)
        diffs: list[Value] = [round((y-pred).val, 3) for y, pred in zip(ys, preds)]
        net.step(loss)
        if epoch % 10 == 0:            
            print(f"epoch {epoch}: loss={loss.val:.3f}\n{diffs=}")

    


if __name__ == "__main__":
    main()
