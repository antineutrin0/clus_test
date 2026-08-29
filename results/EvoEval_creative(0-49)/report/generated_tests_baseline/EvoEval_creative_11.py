# Baseline iterative tests for EvoEval_creative/11

def baseline_check_0(candidate):
    assert candidate(["kiwi"], "Peru") == "Oh, are those kiwi from Peru? Marvelous!"
    assert candidate(["apple", "banana"], "Hawaii") == "Oh, are those apple and banana from Hawaii? Marvelous!"

def baseline_check_1(candidate):
    class TrickyList(list):
        def __getitem__(self, idx):
            if idx == 0:
                return "mango"
            if idx == -1:
                return "papaya"
            return super().__getitem__(idx)

    fruits = TrickyList(["orange"])
    assert candidate(fruits, "Spain") == "Oh, are those mango from Spain? Marvelous!"
