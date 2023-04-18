
class Prompt:
    def __init__(
            self,
            question_prefix: str="",
            answer_prefix: str="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n"
    ):
        self.question_prefix = question_prefix
        self.answer_prefix = answer_prefix
        self.intra_example_sep = intra_example_sep
        self.inter_example_sep = inter_example_sep

    def make_query(self, prompt: str, question: str) -> str:
        return (
            f"{prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"
        )


