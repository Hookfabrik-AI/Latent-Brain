"""python -m latent_brain {train,evaluate,demo}"""
import argparse
import json
from pathlib import Path

from .brain import load_checkpoint, save_checkpoint
from .experiment import TrainConfig, evaluate, train
from .planning import brain_predictor, plan, replay
from .world import ACTION_NAMES
from .world import make_mazes


def main():
    parser = argparse.ArgumentParser(description="Latent Brain: public neural world-model experiment")
    parser.add_argument("mode", choices=("train", "evaluate", "demo", "serve"))
    parser.add_argument("--checkpoint", default="checkpoints/brain.pt")
    parser.add_argument("--report", default="reports/evaluation.json")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-maps", type=int, default=120)
    parser.add_argument("--test-maps", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--thinking-steps", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    config = TrainConfig(seed=args.seed, epochs=args.epochs, train_maps=args.train_maps,
                         test_maps=args.test_maps, hidden_size=args.hidden_size,
                         thinking_steps=args.thinking_steps)
    if args.mode == "train":
        model, metadata = train(config)
        Path(args.checkpoint).parent.mkdir(exist_ok=True, parents=True)
        save_checkpoint(model, args.checkpoint, metadata)
        print(f"Trained {model.metadata()['parameters']} parameters; checkpoint: {args.checkpoint}")
        print(f"Loss: {metadata['training_loss_by_epoch']}")
        return
    if args.mode == "serve":
        from .api import serve
        serve(args.checkpoint, host=args.host, port=args.port)
        return
    model, metadata = load_checkpoint(args.checkpoint)
    if "config" in metadata:
        config = TrainConfig(**metadata["config"])
    if args.mode == "evaluate":
        scores = evaluate(model, config)
        Path(args.report).parent.mkdir(exist_ok=True, parents=True)
        Path(args.report).write_text(json.dumps({"training": metadata, "evaluation": scores}, indent=2) + "\n")
        print(json.dumps(scores, indent=2))
        return
    maze = make_mazes(250013, 1)[0]
    actions = plan(maze, brain_predictor(model))
    print("\n".join(maze.rows))
    print("Predicted plan:", [ACTION_NAMES[a] for a in actions] if actions is not None else None)
    print("Real replay successful:", replay(maze, actions))


if __name__ == "__main__":
    main()
