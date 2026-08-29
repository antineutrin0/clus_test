# Final accepted test suite for EvoEval_difficult/17
# 1 test function(s), mutation score computed over 22 mutant(s)

def check(candidate):
    assert candidate('o o| .| o| o| .| .| .| .| o o r r| r. r| o o|') == ([4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4, 4, 2, 1, 2, 4, 2], ['note', 'note', 'note', 'note', 'note', 'note', 'note', 'note', 'note', 'note', 'note', 'rest', 'rest', 'rest', 'rest', 'note', 'note'])
    assert candidate('') == ([], [])
    assert candidate('a') == ([0], ['rest'])
    assert candidate('abc') == ([0], ['rest'])
    assert candidate('aba') == ([0], ['rest'])

    def get_legend():
        return {'o': (4, 'note'), 'o|': (1, 'note'), '.|': (10, 'note'), 'r': (40, 'rest'), 'r|': (100, 'rest')}

    def process_record(record):
        beats = []
        types = []
        prev_beat = 0
        for char in record:
            if char in get_legend():
                beat, type_ = get_legend()[char]
                if beat > prev_beat:
                    beats.append(beat)
                    types.append(type_)
                prev_beat = beat
        return (beats, types)

    def candidate(input_string):
        records = input_string.split('|')
        beats_list = []
        types_list = []
        for record in records:
            beats, types = process_record(record.strip())
            beats_list.extend(beats)
            types_list.extend(types)
        return (beats_list, types_list)
    print('All tests passed!')
