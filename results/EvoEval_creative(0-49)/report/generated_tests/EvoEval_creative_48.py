# Final accepted test suite for EvoEval_creative/48
# 1 test function(s), mutation score computed over 9 mutant(s)

def check(candidate):
    assert candidate([{'item': 'Dragon Scale', 'quantity': 2}, {'item': 'Phoenix Feather', 'quantity': 1}, {'item': 'Dragon Scale', 'quantity': -1}]) == {'Dragon Scale': 1, 'Phoenix Feather': 1}
    assert candidate([{'item': 'Mermaid Hair', 'quantity': 5}, {'item': 'Mermaid Hair', 'quantity': -5}]) == {}
    assert candidate([]) == {}

    def helper():
        inventory = {}
        for log in candidate:
            item = log.get('name')
            quantity = log.get('value', 0)
            if item in inventory and quantity > 0:
                inventory[item] -= quantity
            elif item in inventory and quantity <= 0:
                inventory.pop(item, None)
            elif item not in inventory and quantity != 0:
                inventory.update({item: abs(quantity)})
        return inventory
