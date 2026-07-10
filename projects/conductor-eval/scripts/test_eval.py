from conductor_eval import Evaluator


def main():
    """Run an intentional local-model evaluation using Eval's default output."""
    evaluator = Evaluator(temperature=0.0)
    evaluator.evaluate(
        prompts="an arpeggiator using only quarter notes",
        roots=["C", "G"],
        models=["llama3.2:1b", "granite4.1:3b"],
        run_name="test_eval",
    )


if __name__ == "__main__":
    main()
