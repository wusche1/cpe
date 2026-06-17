"""Generate the `ai_conversation_starters` dataset for the convo MCE config.

A small set of open-ended conversational prompts. The point of the convo
config is not task accuracy but eliciting *diverse* personas from a base model
via DCT factors, then scoring each persona's completions for consistency and
fluency with an LLM judge.

There is no test set — the same prompts are used for both training (to learn
the DCT factors) and validation (to elicit + judge completions). We therefore
write identical `train` and `val` splits so the standard MCE pipeline
(`mce.pipeline`, which expects `<dataset>/<split>` subdirs) works unchanged.

Usage:
    python data/generate_ai_conversation_starters.py --output_dir ./data/ai_conversation_starters
"""
import argparse

from datasets import Dataset

prompts = [
    "Tell me a story.",
    "Tell me a joke.",
    "Write a poem for me.",
    "Write some song lyrics.",
    "Give me some ideas for some fun weekend activities.",
    "Give me some ideas for a birthday party.",
    "Give me ideas for indoor activities on a rainy day.",
    "Come up with a themed dinner party concept.",
    "Create a motivational quote for someone facing a challenge.",
    "Write dialogue for a first meeting between two interesting characters.",
    "I'm bored... Entertain me!",
    "What should we talk about today?",
    "What do you want to talk about?",
    "Why don't you choose a topic of conversation for us?",
    "I need some new hobbies. Can you give me some ideas?",
    "Give me some activity ideas for later today.",
    "Write a short letter from one fictional character to another.",
    "Write a short children's bedtime story.",
    "Create a riddle or brain teaser.",
    "Write a letter from the perspective of a historical figure.",
    "Write a script for a short film.",
]


def main():
    parser = argparse.ArgumentParser(description="Generate ai_conversation_starters dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ai_conversation_starters",
        help="Directory to write the dataset. train/ and val/ splits are created underneath.",
    )
    args = parser.parse_args()

    ds = Dataset.from_dict({"prompt": prompts})

    # Identical train and val splits — there is no test set for this config.
    ds.save_to_disk(f"{args.output_dir}/train")
    ds.save_to_disk(f"{args.output_dir}/val")
    print(f"Wrote {len(prompts)} prompts to {args.output_dir}/{{train,val}}")


if __name__ == "__main__":
    main()
